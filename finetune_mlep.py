"""
Callable notebook-friendly fine-tuning for MLEP.

Put this file beside:
    - mlep_fast_evaluator.py
    - utils.py
    - MLEP/
    - Datasets/

Notebook usage:
    from finetune_mlep_callable import finetune_mlep

    result = finetune_mlep(
        dataset_name="gravex200k",
        weights_path="MLEP/pretrained/model_epoch_best.pth",
        max_per_class=5000,
        epochs=3,
        batch_size=32,
        lr=1e-5,
        num_workers=0,
    )
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from utils import get_dataset_roots, collect_images, compute_metrics
from mlep_fast_evaluator import load_mlep_model, get_mlep_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MLEPTrainDataset(Dataset):
    def __init__(self, samples: List[dict], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(float(sample["true_label"]), dtype=torch.float32)
        return {
            "image": image,
            "label": label,
            "id": sample["id"],
        }


def collate_train(batch: List[dict]) -> dict:
    return {
        "images": torch.stack([x["image"] for x in batch], dim=0),
        "labels": torch.stack([x["label"] for x in batch], dim=0),
        "ids": [x["id"] for x in batch],
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def pick_dataset(dataset_name: Optional[str]) -> Tuple[str, Path]:
    roots = get_dataset_roots()
    if not roots:
        raise RuntimeError("get_dataset_roots() returned no datasets.")

    if dataset_name is None or str(dataset_name).lower() == "auto":
        preferred = [
            "20K_real_and_deepfake_images",
            "Deepfake-vs-Real-v2",
            "DeepDetect-2025",
            "nuriachandra_Deepfake-Eval-2024",
            "gravex200k",
        ]
        for name in preferred:
            if name in roots and Path(roots[name]).exists():
                return name, Path(roots[name])

        for name, root in roots.items():
            if Path(root).exists():
                return name, Path(root)
        raise RuntimeError("No dataset root exists on disk.")

    if dataset_name not in roots:
        available = "\n".join([f"- {x}" for x in roots.keys()])
        raise ValueError(f"Unknown dataset_name={dataset_name!r}. Available:\n{available}")

    root = Path(roots[dataset_name])
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return dataset_name, root


def balanced_subset(samples: List[dict], max_per_class: Optional[int], seed: int) -> List[dict]:
    by_label = {0: [], 1: []}
    for s in samples:
        y = int(s["true_label"])
        if y in by_label:
            by_label[y].append(s)

    rng = random.Random(seed)
    selected = []
    for label in [0, 1]:
        items = by_label[label]
        rng.shuffle(items)
        if max_per_class is not None and max_per_class > 0:
            items = items[:max_per_class]
        selected.extend(items)

    rng.shuffle(selected)
    return selected


def train_val_split(samples: List[dict], val_fraction: float, seed: int) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    train_samples = []
    val_samples = []

    for label in [0, 1]:
        items = [s for s in samples if int(s["true_label"]) == label]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction)) if len(items) > 1 else 0
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def count_labels(samples: List[dict]) -> Dict[int, int]:
    counts = {0: 0, 1: 0}
    for s in samples:
        counts[int(s["true_label"])] += 1
    return counts


def maybe_freeze_backbone(model: nn.Module) -> None:
    trainable_keywords = ["fc", "classifier", "head", "linear", "last"]
    found = False

    for name, param in model.named_parameters():
        should_train = any(k in name.lower() for k in trainable_keywords)
        param.requires_grad = should_train
        found = found or should_train

    if not found:
        print("Could not identify classifier/head parameters. Training all parameters instead.", flush=True)
        for param in model.parameters():
            param.requires_grad = True


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    scaler=None,
    train: bool = True,
    threshold: float = 0.5,
):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_count = 0
    tp = tn = fp = fn = 0

    desc = "train" if train else "val"
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in tqdm(loader, desc=desc, dynamic_ncols=True, leave=True, position=0,):
            images = batch["images"].to(device, non_blocking=True).float()
            labels = batch["labels"].to(device, non_blocking=True).float()

            if train:
                optimizer.zero_grad(set_to_none=True)

            use_amp = scaler is not None and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images).view(-1)
                loss = criterion(logits, labels)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.numel()
            total_loss += float(loss.detach().cpu()) * batch_size
            total_count += batch_size

            scores = torch.sigmoid(logits.detach())
            preds = (scores >= threshold).long().cpu().tolist()
            y_true = labels.detach().long().cpu().tolist()

            for yt, yp in zip(y_true, preds):
                if yt == 1 and yp == 1:
                    tp += 1
                elif yt == 0 and yp == 0:
                    tn += 1
                elif yt == 0 and yp == 1:
                    fp += 1
                elif yt == 1 and yp == 0:
                    fn += 1

    avg_loss = total_loss / max(1, total_count)
    metrics = compute_metrics(tp, tn, fp, fn)
    return avg_loss, {"TP": tp, "TN": tn, "FP": fp, "FN": fn}, metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: dict,
    dataset_name: str,
    epoch: int,
    val_loss: float,
    val_metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "dataset_name": dataset_name,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "config": config,
        },
        path,
    )


def finetune_mlep(
    dataset_name: str = "20K_real_and_deepfake_images",
    repo_dir: str = "MLEP",
    weights_path: str = "MLEP/pretrained/model_epoch_best.pth",
    output_dir: str = "Finetuned/MLEP",
    max_per_class: int = 1000,
    val_fraction: float = 0.2,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    seed: int = 42,
    threshold: float = 0.5,
    freeze_backbone: bool = False,
    load_size: int = 256,
    crop_size: int = 224,
    no_resize: bool = False,
    center_crop: bool = False,
    amp: bool = False,
    return_model: bool = False,
):
    """
    Notebook-friendly MLEP fine-tuning function.

    Returns a dict containing paths and training history. If return_model=True,
    it also includes the trained model and device.
    """
    print("Starting MLEP fine-tuning...", flush=True)
    set_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    dataset_name, dataset_root = pick_dataset(dataset_name)
    print("Dataset:", dataset_name, flush=True)
    print("Dataset root:", dataset_root, flush=True)

    print("Collecting images...", flush=True)
    samples = collect_images(dataset_root)
    if not samples:
        raise RuntimeError(f"No images found in {dataset_root}")

    max_per_class_or_none = None if max_per_class == 0 else max_per_class
    samples = balanced_subset(samples, max_per_class=max_per_class_or_none, seed=seed)
    train_samples, val_samples = train_val_split(samples, val_fraction=val_fraction, seed=seed)

    print("Total selected:", len(samples), count_labels(samples), flush=True)
    print("Train:", len(train_samples), count_labels(train_samples), flush=True)
    print("Val:", len(val_samples), count_labels(val_samples), flush=True)

    transform = get_mlep_transform(
        load_size=load_size,
        crop_size=crop_size,
        no_resize=no_resize,
        no_crop=not center_crop,
    )

    train_ds = MLEPTrainDataset(train_samples, transform=transform)
    val_ds = MLEPTrainDataset(val_samples, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_train,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_train,
        persistent_workers=(num_workers > 0),
    )

    print("Loading MLEP model...", flush=True)
    model, device = load_mlep_model(
        repo_dir=repo_dir,
        weights_path=weights_path,
        device=str(device),
    )

    if freeze_backbone:
        maybe_freeze_backbone(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameters:", sum(p.numel() for p in trainable_params), flush=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    output_dir_path = Path(output_dir)
    best_path = output_dir_path / f"MLEP_finetuned_{dataset_name}_best.pth"
    last_path = output_dir_path / f"MLEP_finetuned_{dataset_name}_last.pth"
    history_path = output_dir_path / f"MLEP_finetuned_{dataset_name}_history.json"

    config = {
        "dataset_name": dataset_name,
        "repo_dir": repo_dir,
        "weights_path": weights_path,
        "output_dir": output_dir,
        "max_per_class": max_per_class,
        "val_fraction": val_fraction,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "seed": seed,
        "threshold": threshold,
        "freeze_backbone": freeze_backbone,
        "load_size": load_size,
        "crop_size": crop_size,
        "no_resize": no_resize,
        "center_crop": center_crop,
        "amp": amp,
    }

    best_val_acc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}", flush=True)
        train_loss, train_counts, train_metrics = run_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler, train=True, threshold=threshold
        )
        val_loss, val_counts, val_metrics = run_one_epoch(
            model, val_loader, optimizer, criterion, device, scaler=None, train=False, threshold=threshold
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_counts": train_counts,
            "train_metrics": train_metrics,
            "val_loss": val_loss,
            "val_counts": val_counts,
            "val_metrics": val_metrics,
        }
        history.append(row)

        print("Train loss:", round(train_loss, 6), "metrics:", train_metrics, "counts:", train_counts, flush=True)
        print("Val   loss:", round(val_loss, 6), "metrics:", val_metrics, "counts:", val_counts, flush=True)

        save_checkpoint(last_path, model, config, dataset_name, epoch, val_loss, val_metrics)

        val_acc = float(val_metrics.get("accuracy", 0.0))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_path, model, config, dataset_name, epoch, val_loss, val_metrics)
            print(f"Saved new best checkpoint: {best_path}", flush=True)

        output_dir_path.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("\nDone.", flush=True)
    print("Best checkpoint:", best_path, flush=True)
    print("Last checkpoint:", last_path, flush=True)
    print("History:", history_path, flush=True)

    result = {
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_path": str(history_path),
        "history": history,
    }
    if return_model:
        result["model"] = model
        result["device"] = device
    return result
