from pathlib import Path
import json
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm.auto import tqdm

from utils import get_dataset_roots, collect_images, compute_metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = PROJECT_ROOT / "MLEP"


def _prepare_repo_imports(repo_dir: Path):
    repo_dir = Path(repo_dir).resolve()
    if not repo_dir.exists():
        import subprocess
        print(f"Cloning MLEP repo into {repo_dir}...")
        subprocess.run(
            ["git", "clone", "https://github.com/fkeufss/MLEP.git", str(repo_dir)],
            check=True,
        )
        print("MLEP repo cloned successfully.")

    for name in list(sys.modules.keys()):
        if name == "networks" or name.startswith("networks."):
            del sys.modules[name]

    if str(repo_dir) in sys.path:
        sys.path.remove(str(repo_dir))
    sys.path.insert(0, str(repo_dir))


# ---------- dataset ----------
class ImageEvalDatasetMLEP(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "id": sample["id"],
            "split": sample["split"],
            "true_label": int(sample["true_label"]),
        }


def collate_mlep(batch):
    return {
        "images": torch.stack([x["image"] for x in batch], dim=0),
        "ids": [x["id"] for x in batch],
        "splits": [x["split"] for x in batch],
        "true_labels": [int(x["true_label"]) for x in batch],
    }


# ---------- preprocessing ----------
def get_mlep_transform(load_size=256, crop_size=224, no_resize=False, no_crop=True):
    steps = []
    if not no_resize:
        steps.append(transforms.Resize((load_size, load_size)))
    if not no_crop:
        steps.append(transforms.CenterCrop(crop_size))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(steps)


# ---------- model loading ----------
def load_mlep_model(
    repo_dir=None,
    weights_path=None,
    device=None,
):
    """
    Load MLEP model (modified ResNet-50 with Multi-scale Local Entropy).

    Expected weights: MLEP/pretrained/model_epoch_best.pth
    """
    repo_dir = Path(repo_dir) if repo_dir is not None else DEFAULT_REPO_DIR
    _prepare_repo_imports(repo_dir)

    from networks.resnet import resnet50

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = resnet50(num_classes=1)

    if weights_path is None:
        weights_path = repo_dir / "pretrained" / "model_epoch_best.pth"
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find weights file: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Remove 'module.' prefix if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "", 1) if k.startswith("module.") else k
        clean_state_dict[new_k] = v

    model.load_state_dict(clean_state_dict, strict=True)
    print(f"Loaded MLEP model weights: {weights_path}")

    model.to(device)
    model.eval()
    return model, device


# ---------- prediction ----------
@torch.no_grad()
def predict_dataloader_mlep(
    model,
    dataloader,
    device,
    checkpoint_file=None,
    checkpoint_every=100,
    threshold=0.5,
    invert_labels=False,
    desc="",
):
    """
    MLEP outputs a single logit per image.

    Default:
        sigmoid(logit) >= threshold => fake (1)

    If invert_labels=True:
        sigmoid(logit) >= threshold => real (0)
    """
    model.eval()
    buffer = []

    for batch_idx, batch in enumerate(
        tqdm(
            dataloader,
            desc=desc,
            dynamic_ncols=True,
            mininterval=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ),
        start=1,
    ):
        images = batch["images"].to(device, non_blocking=True).float()
        logits = model(images)
        fake_scores = torch.sigmoid(logits).detach().float().cpu().view(-1).tolist()

        for sample_id, split, true_label, fake_score in zip(
            batch["ids"], batch["splits"], batch["true_labels"], fake_scores
        ):
            raw_pred_label = 1 if fake_score >= threshold else 0
            pred_label = 1 - raw_pred_label if invert_labels else raw_pred_label

            buffer.append({
                "id": sample_id,
                "split": split,
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "raw_pred_label": int(raw_pred_label),
                "fake_score": float(fake_score),
                "invert_labels": bool(invert_labels),
            })

        if checkpoint_file is not None and batch_idx % checkpoint_every == 0:
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                for row in buffer:
                    f.write(json.dumps(row) + "\n")
                f.flush()
            buffer = []

    return buffer


def _load_checkpoint_rows(checkpoint_file):
    rows = []
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _build_result(rows, model_name, dataset_name, dataset_root, threshold, invert_labels=False):
    tp = tn = fp = fn = 0
    for row in rows:
        y_true = int(row["true_label"])
        y_pred = int(row["pred_label"])
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "num_samples": len(rows),
        "threshold": threshold,
        "invert_labels": bool(invert_labels),
        "counts": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "metrics": compute_metrics(tp, tn, fp, fn),
        "misclassified_ids": {
            "FP": [x["id"] for x in rows if int(x["true_label"]) == 0 and int(x["pred_label"]) == 1],
            "FN": [x["id"] for x in rows if int(x["true_label"]) == 1 and int(x["pred_label"]) == 0],
        },
    }


# ---------- evaluation ----------
def evaluate_dataset_mlep_fast(
    dataset_name,
    dataset_root,
    model,
    device,
    model_name="MLEP_ResNet50",
    batch_size=64,
    checkpoint_every=100,
    num_workers=0,
    max_batches=None,
    threshold=0.5,
    invert_labels=False,
    load_size=256,
    crop_size=224,
    no_resize=False,
    no_crop=True,
    progress_prefix="",
):
    results_dir = Path("Results")
    checkpoint_dir = results_dir / "_checkpoints"
    results_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    result_file = results_dir / f"{model_name}_{dataset_name}.json"
    checkpoint_file = checkpoint_dir / f"{model_name}_{dataset_name}.jsonl"

    samples = collect_images(dataset_root)

    done_ids = set()
    if checkpoint_file.exists():
        for row in _load_checkpoint_rows(checkpoint_file):
            done_ids.add(row["id"])
        print(f"{progress_prefix}Resuming {dataset_name}: {len(done_ids)} already processed")

    remaining = [x for x in samples if x["id"] not in done_ids]
    if max_batches is not None:
        remaining = remaining[: batch_size * max_batches]

    print(f"{progress_prefix}Running {dataset_name}: {len(remaining)} remaining")

    transform = get_mlep_transform(
        load_size=load_size, crop_size=crop_size,
        no_resize=no_resize, no_crop=no_crop,
    )
    dataset = ImageEvalDatasetMLEP(remaining, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_mlep,
        persistent_workers=(num_workers > 0),
    )

    tail_buffer = predict_dataloader_mlep(
        model=model,
        dataloader=dataloader,
        device=device,
        checkpoint_file=checkpoint_file,
        checkpoint_every=checkpoint_every,
        threshold=threshold,
        invert_labels=invert_labels,
        desc=f"{progress_prefix}{dataset_name}",
    )

    if tail_buffer:
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            for row in tail_buffer:
                f.write(json.dumps(row) + "\n")
            f.flush()

    rows = _load_checkpoint_rows(checkpoint_file)
    result = _build_result(
        rows=rows,
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        threshold=threshold,
        invert_labels=invert_labels,
    )

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def evaluate_all_datasets_mlep_fast(
    model,
    device,
    model_name="MLEP_ResNet50",
    batch_size=64,
    checkpoint_every=100,
    num_workers=0,
    max_batches=None,
    threshold=0.5,
    invert_labels=False,
    load_size=256,
    crop_size=224,
    no_resize=False,
    no_crop=True,
):
    dataset_roots = get_dataset_roots()
    all_results = {}

    items = list(dataset_roots.items())
    total = len(items)

    for idx, (dataset_name, dataset_root) in enumerate(items, start=1):
        prefix = f"[{idx}/{total}] "
        print(f"\n{prefix}Running dataset: {dataset_name}")
        all_results[dataset_name] = evaluate_dataset_mlep_fast(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            model=model,
            device=device,
            model_name=model_name,
            batch_size=batch_size,
            checkpoint_every=checkpoint_every,
            num_workers=num_workers,
            max_batches=max_batches,
            threshold=threshold,
            invert_labels=invert_labels,
            load_size=load_size,
            crop_size=crop_size,
            no_resize=no_resize,
            no_crop=no_crop,
            progress_prefix=prefix,
        )

    return all_results


if __name__ == "__main__":
    model, device = load_mlep_model(
        repo_dir="MLEP",
        weights_path="MLEP/pretrained/model_epoch_best.pth",
    )

    results = evaluate_all_datasets_mlep_fast(
        model=model,
        device=device,
        model_name="MLEP_ResNet50",
        batch_size=64,
        checkpoint_every=50,
        num_workers=0,
        max_batches=3,
        threshold=0.5,
        invert_labels=False,
    )

    print(json.dumps(results, indent=2))
