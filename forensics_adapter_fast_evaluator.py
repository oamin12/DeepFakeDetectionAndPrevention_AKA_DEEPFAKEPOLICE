from pathlib import Path
import json
import sys
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm.auto import tqdm

from utils import get_dataset_roots, collect_images, compute_metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_DIR = PROJECT_ROOT / "ForensicsAdapter"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from model.ds import DS


# ---------- config / model loading ----------
def load_forensics_adapter(
    config_path=None,
    weights_path=None,
    device=None,
    clip_download_root=None,
):
    """
    Load the official ForensicsAdapter DS model.

    Expected repo layout:
        project_root/
        ├── ForensicsAdapter/
        │   ├── config/test.yaml
        │   ├── model/ds.py
        │   └── ...
        ├── forensics_adapter_fast_evaluator.py
        ├── utils.py
        └── Datasets/

    Put the downloaded official weights anywhere, then pass weights_path.
    """
    if config_path is None:
        config_path = REPO_DIR / "config" / "test.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # The repo config may hardcode cuda:0. Keep it aligned with the real device.
    config["device"] = str(device)
    config["cuda"] = device.type == "cuda"

    # Optional patch: official ds.py hardcodes download_root='/data/cuixinjie/weights'.
    # If you edited ds.py to accept a custom root, this field can be used there.
    if clip_download_root is not None:
        config["clip_download_root"] = str(clip_download_root)

    model = DS(
        clip_name=config["clip_model_name"],
        adapter_vit_name=config["vit_name"],
        num_quires=config["num_quires"],
        fusion_map=config["fusion_map"],
        mlp_dim=config["mlp_dim"],
        mlp_out_dim=config["mlp_out_dim"],
        head_num=config["head_num"],
        device=str(device),
    )

    if weights_path is None:
        weights_path = config.get("weights_path")
    if not weights_path:
        raise ValueError("Pass weights_path to the official ForensicsAdapter .pth checkpoint.")

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find weights file: {weights_path}")

    ckpt = torch.load(weights_path, map_location=device)

    # Handle common checkpoint formats: raw state_dict, {'state_dict': ...}, {'model': ...}
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
    else:
        state_dict = ckpt

    # Handle DataParallel prefixes if present.
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "", 1) if k.startswith("module.") else k
        clean_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    print(f"Loaded ForensicsAdapter weights: {weights_path}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model, device, config


# ---------- image preprocessing ----------
def get_forensics_adapter_transform(config):
    size = int(config.get("resolution", 256))
    mean = config.get("mean", [0.48145466, 0.4578275, 0.40821073])
    std = config.get("std", [0.26862954, 0.26130258, 0.27577711])

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class ImageEvalDatasetFA(Dataset):
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


def collate_forensics_adapter(batch, config):
    images = torch.stack([x["image"] for x in batch], dim=0)
    labels = torch.tensor([x["true_label"] for x in batch], dtype=torch.long)
    ids = [x["id"] for x in batch]
    splits = [x["split"] for x in batch]

    bsz = images.shape[0]
    resolution = int(config.get("resolution", 256))

    # The official test dataset returns these fields even during inference.
    # For external image folders we do not have masks/landmarks, so we pass safe zero tensors.
    mask = torch.zeros((bsz, resolution, resolution, 1), dtype=torch.float32)
    xray = torch.zeros((bsz, 1, resolution, resolution), dtype=torch.float32)
    landmark = None

    # DS.forward accesses if_boundary directly. For 224/16 = 14 => 196 patch flags.
    patch_label = torch.zeros((bsz, 196), dtype=torch.long)
    clip_patch_label = torch.zeros((bsz, 256), dtype=torch.long)  # 224/14 = 16 => 256
    if_boundary = torch.zeros((bsz, 196), dtype=torch.long)

    data_dict = {
        "image": images,
        "label": labels,
        "mask": mask,
        "landmark": landmark,
        "xray": xray,
        "patch_label": patch_label,
        "clip_patch_label": clip_patch_label,
        "if_boundary": if_boundary,
    }

    return {
        "data_dict": data_dict,
        "ids": ids,
        "splits": splits,
        "true_labels": labels.tolist(),
    }


# ---------- prediction ----------
@torch.no_grad()
def predict_dataloader_forensics_adapter(
    model,
    dataloader,
    device,
    checkpoint_file=None,
    checkpoint_every=100,
    threshold=0.5,
    desc="",
):
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
        data_dict = {}
        for k, v in batch["data_dict"].items():
            if torch.is_tensor(v):
                data_dict[k] = v.to(device, non_blocking=True)
            else:
                data_dict[k] = v

        out = model(data_dict, inference=True)

        # Official DS.forward returns prob = softmax(cls)[:, 1], i.e. fake probability.
        fake_scores = out["prob"].detach().float().cpu().view(-1).tolist()

        for sample_id, split, true_label, fake_score in zip(
            batch["ids"], batch["splits"], batch["true_labels"], fake_scores
        ):
            pred_label = 1 if fake_score >= threshold else 0
            buffer.append({
                "id": sample_id,
                "split": split,
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "fake_score": float(fake_score),
            })

        if checkpoint_file is not None and batch_idx % checkpoint_every == 0:
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                for row in buffer:
                    f.write(json.dumps(row) + "\n")
                f.flush()
            buffer = []

    return buffer


# ---------- evaluation ----------
def evaluate_dataset_forensics_adapter_fast(
    dataset_name,
    dataset_root,
    model,
    device,
    config,
    model_name="ForensicsAdapter",
    batch_size=32,
    checkpoint_every=100,
    num_workers=4,
    max_batches=None,
    threshold=0.5,
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
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])

    remaining = [x for x in samples if x["id"] not in done_ids]
    print(f"{progress_prefix}Running {dataset_name}: {len(remaining)} remaining")

    if max_batches is not None:
        remaining = remaining[: batch_size * max_batches]

    transform = get_forensics_adapter_transform(config)
    dataset = ImageEvalDatasetFA(remaining, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: collate_forensics_adapter(b, config),
        persistent_workers=(num_workers > 0),
    )

    tail_buffer = predict_dataloader_forensics_adapter(
        model=model,
        dataloader=dataloader,
        device=device,
        checkpoint_file=checkpoint_file,
        checkpoint_every=checkpoint_every,
        threshold=threshold,
        desc=f"{progress_prefix}{dataset_name}",
    )

    if tail_buffer:
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            for row in tail_buffer:
                f.write(json.dumps(row) + "\n")
            f.flush()

    rows = []
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    tp = tn = fp = fn = 0
    for row in rows:
        y_true = row["true_label"]
        y_pred = row["pred_label"]

        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        else:
            fn += 1

    result = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "num_samples": len(rows),
        "threshold": threshold,
        "counts": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "metrics": compute_metrics(tp, tn, fp, fn),
        "misclassified_ids": {
            "FP": [x["id"] for x in rows if x["true_label"] == 0 and x["pred_label"] == 1],
            "FN": [x["id"] for x in rows if x["true_label"] == 1 and x["pred_label"] == 0],
        },
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def evaluate_all_datasets_forensics_adapter_fast(
    model,
    device,
    config,
    model_name="ForensicsAdapter",
    batch_size=32,
    checkpoint_every=100,
    num_workers=4,
    max_batches=None,
    threshold=0.5,
):
    dataset_roots = get_dataset_roots()
    all_results = {}

    items = list(dataset_roots.items())
    total = len(items)

    for idx, (dataset_name, dataset_root) in enumerate(items, start=1):
        prefix = f"[{idx}/{total}] "
        print(f"\n{prefix}Running dataset: {dataset_name}")
        all_results[dataset_name] = evaluate_dataset_forensics_adapter_fast(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            model=model,
            device=device,
            config=config,
            model_name=model_name,
            batch_size=batch_size,
            checkpoint_every=checkpoint_every,
            num_workers=num_workers,
            max_batches=max_batches,
            threshold=threshold,
            progress_prefix=prefix,
        )

    return all_results


if __name__ == "__main__":
    # Example smoke run. Edit weights_path first.
    model, device, config = load_forensics_adapter(
        config_path=REPO_DIR / "config" / "test.yaml",
        weights_path=REPO_DIR / "weights" / "ckpt_best.pth",
    )

    results = evaluate_all_datasets_forensics_adapter_fast(
        model=model,
        device=device,
        config=config,
        model_name="ForensicsAdapter",
        batch_size=16,
        checkpoint_every=50,
        num_workers=0,   # safer on Windows/Jupyter; increase later if stable
        max_batches=2,   # remove after smoke test
        threshold=0.5,
    )

    print(json.dumps(results, indent=2))
