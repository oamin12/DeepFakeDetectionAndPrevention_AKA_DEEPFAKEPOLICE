from pathlib import Path
import json
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

from utils import get_dataset_roots, collect_images, compute_metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = PROJECT_ROOT / "FSFM-CVPR25" / "fsfm-3c"

# HuggingFace model info
HF_REPO_ID = "Wolowolo/fsfm-3c"
DEFAULT_CKPT_NAME = "finetuned_models/FF++_c23_32frames/checkpoint-min_val_loss.pth"


def _prepare_repo_imports(repo_dir: Path):
    """
    FSFM repo contains models_vit.py needed to build the architecture.
    Clone from: https://github.com/wolo-wolo/FSFM-CVPR25
    """
    repo_dir = Path(repo_dir).resolve()
    if not repo_dir.exists():
        import subprocess
        # repo_dir points to FSFM-CVPR25/fsfm-3c, so clone the parent
        parent = repo_dir.parent
        if not parent.exists():
            print(f"Cloning FSFM-CVPR25 repo into {parent}...")
            subprocess.run(
                ["git", "clone", "https://github.com/wolo-wolo/FSFM-CVPR25.git", str(parent)],
                check=True,
            )
            print("FSFM-CVPR25 repo cloned successfully.")

    for name in list(sys.modules.keys()):
        if name == "models_vit" or name.startswith("models_vit."):
            del sys.modules[name]

    if str(repo_dir) in sys.path:
        sys.path.remove(str(repo_dir))
    sys.path.insert(0, str(repo_dir))


# ---------- dataset ----------
class ImageEvalDatasetFSFM(Dataset):
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


def collate_fsfm(batch):
    return {
        "images": torch.stack([x["image"] for x in batch], dim=0),
        "ids": [x["id"] for x in batch],
        "splits": [x["split"] for x in batch],
        "true_labels": [int(x["true_label"]) for x in batch],
    }


# ---------- preprocessing ----------
def get_fsfm_transform(image_size=224, mean=None, std=None):
    """
    FSFM uses 224x224 with dataset-specific normalization.
    If mean/std not provided, falls back to ImageNet defaults.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_pretrain_mean_std(repo_dir=None, ckpt_dir=None):
    """
    Try to load dataset-specific mean/std from pretrain_ds_mean_std.txt
    shipped with FSFM checkpoints. Falls back to ImageNet defaults.
    """
    candidates = []
    if ckpt_dir:
        candidates.append(Path(ckpt_dir) / "pretrain_ds_mean_std.txt")
    if repo_dir:
        candidates.append(Path(repo_dir) / "pretrain_ds_mean_std.txt")

    for path in candidates:
        if path.exists():
            with open(path, "r") as f:
                lines = f.read().strip().split("\n")
            # Expected format: mean line then std line, each with 3 floats
            if len(lines) >= 2:
                mean = [float(x) for x in lines[0].split()]
                std = [float(x) for x in lines[1].split()]
                if len(mean) == 3 and len(std) == 3:
                    print(f"Loaded FSFM normalization from: {path}")
                    return mean, std

    print("Using ImageNet normalization defaults for FSFM")
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---------- model loading ----------
def load_fsfm_model(
    repo_dir=None,
    weights_path=None,
    hf_ckpt_name=None,
    model_variant="vit_base_patch16",
    device=None,
    download_from_hf=True,
):
    """
    Load FSFM-3C model (ViT-B/16 for deepfake detection).

    If weights_path is provided, loads from that file.
    Otherwise downloads from HuggingFace (Wolowolo/fsfm-3c).
    """
    repo_dir = Path(repo_dir) if repo_dir is not None else DEFAULT_REPO_DIR
    _prepare_repo_imports(repo_dir)

    import models_vit

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = models_vit.__dict__[model_variant](
        num_classes=2,
        drop_path_rate=0.1,
        global_pool=True,
    )

    # Patch forward to avoid newer timm passing unexpected kwargs
    _orig_forward_features = model.forward_features

    def _patched_forward(x, **kwargs):
        x = _orig_forward_features(x)
        x = model.head(x)
        return x

    model.forward = _patched_forward

    if weights_path is None and download_from_hf:
        ckpt_name = hf_ckpt_name or DEFAULT_CKPT_NAME
        local_dir = str(PROJECT_ROOT / "fsfm_weights")
        print(f"Downloading FSFM checkpoint from HF: {HF_REPO_ID}/{ckpt_name}")
        weights_path = hf_hub_download(
            local_dir=local_dir,
            repo_id=HF_REPO_ID,
            filename=ckpt_name,
        )
    elif weights_path is None:
        weights_path = PROJECT_ROOT / "fsfm_weights" / DEFAULT_CKPT_NAME

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find weights file: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Remove 'module.' prefix if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "", 1) if k.startswith("module.") else k
        clean_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    print(f"Loaded FSFM model weights: {weights_path}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model, device


# ---------- prediction ----------
@torch.no_grad()
def predict_dataloader_fsfm(
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
    FSFM outputs 2-class logits [real, fake]. We use softmax[:,1] as fake_score.
    If invert_labels=True, the final predicted label is flipped.
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
        probs = F.softmax(logits, dim=1)
        fake_scores = probs[:, 1].detach().float().cpu().tolist()

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
                "fake_score": float(fake_score),
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
        "invert_labels": invert_labels,
        "counts": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "metrics": compute_metrics(tp, tn, fp, fn),
        "misclassified_ids": {
            "FP": [x["id"] for x in rows if int(x["true_label"]) == 0 and int(x["pred_label"]) == 1],
            "FN": [x["id"] for x in rows if int(x["true_label"]) == 1 and int(x["pred_label"]) == 0],
        },
    }


# ---------- evaluation ----------
def evaluate_dataset_fsfm_fast(
    dataset_name,
    dataset_root,
    model,
    device,
    model_name="FSFM_ViT_B16",
    batch_size=32,
    checkpoint_every=100,
    num_workers=0,
    max_batches=None,
    threshold=0.5,
    invert_labels=False,
    image_size=224,
    mean=None,
    std=None,
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

    transform = get_fsfm_transform(image_size=image_size, mean=mean, std=std)
    dataset = ImageEvalDatasetFSFM(remaining, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fsfm,
        persistent_workers=(num_workers > 0),
    )

    tail_buffer = predict_dataloader_fsfm(
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


def evaluate_all_datasets_fsfm_fast(
    model,
    device,
    model_name="FSFM_ViT_B16",
    batch_size=32,
    checkpoint_every=100,
    num_workers=0,
    max_batches=None,
    threshold=0.5,
    invert_labels=False,
    image_size=224,
    mean=None,
    std=None,
):
    dataset_roots = get_dataset_roots()
    all_results = {}

    items = list(dataset_roots.items())
    total = len(items)

    for idx, (dataset_name, dataset_root) in enumerate(items, start=1):
        prefix = f"[{idx}/{total}] "
        print(f"\n{prefix}Running dataset: {dataset_name}")
        all_results[dataset_name] = evaluate_dataset_fsfm_fast(
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
            image_size=image_size,
            mean=mean,
            std=std,
            progress_prefix=prefix,
        )

    return all_results


if __name__ == "__main__":
    model, device = load_fsfm_model(
        repo_dir="FSFM-CVPR25",
        download_from_hf=True,
        hf_ckpt_name="finetuned_models/FF++_c23_32frames/checkpoint-min_val_loss.pth",
    )

    results = evaluate_all_datasets_fsfm_fast(
        model=model,
        device=device,
        model_name="FSFM_ViT_B16_FFpp_c23",
        batch_size=32,
        checkpoint_every=50,
        num_workers=0,
        max_batches=3,
        threshold=0.5,
        invert_labels=False,
    )

    print(json.dumps(results, indent=2))
