from pathlib import Path
import contextlib
import io
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
DEFAULT_REPO_DIR = PROJECT_ROOT / "Defocus-Deepfake-Detection"


def _prepare_repo_imports(repo_dir: Path):
    """
    Defocus repo uses top-level packages named `model` and `util`.
    Other repos you tested also use `model`, so clear cached modules first.
    Best practice: restart kernel before switching between repos.
    """
    repo_dir = Path(repo_dir).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"Could not find repo folder: {repo_dir}\n"
            "Clone it in your project root using:\n"
            "git clone https://github.com/irissun9602/Defocus-Deepfake-Detection.git"
        )

    for name in list(sys.modules.keys()):
        if name == "model" or name.startswith("model.") or name == "util" or name.startswith("util."):
            del sys.modules[name]

    if str(repo_dir) in sys.path:
        sys.path.remove(str(repo_dir))
    sys.path.insert(0, str(repo_dir))


# ---------- dataset ----------
class ImageEvalDatasetDefocus(Dataset):
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


def collate_defocus(batch):
    return {
        "images": torch.stack([x["image"] for x in batch], dim=0),
        "ids": [x["id"] for x in batch],
        "splits": [x["split"] for x in batch],
        "true_labels": [int(x["true_label"]) for x in batch],
    }


# ---------- preprocessing ----------
def get_defocus_transform(image_size=299):
    # This matches the official training script: Resize(299), ToTensor, Normalize(0.5, 0.5)
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


# ---------- model loading ----------
def load_defocus_model(
    repo_dir=None,
    weights_path=None,
    backbone="legacy_xception",
    device=None,
    strict=False,
):
    """
    Load official DefocusNet model from Defocus-Deepfake-Detection.

    Typical weights:
        Defocus-Deepfake-Detection/weights/defocus_gt_legacy_xception_Deepfakes.pth
        Defocus-Deepfake-Detection/weights/defocus_gt_legacy_xception_Face2Face.pth
        Defocus-Deepfake-Detection/weights/defocus_gt_legacy_xception_FaceSwap.pth
        Defocus-Deepfake-Detection/weights/defocus_gt_legacy_xception_NeuralTextures.pth
        Defocus-Deepfake-Detection/weights/defocus_gt_legacy_xception_cifake.pth

    Returns: model, device
    """
    repo_dir = Path(repo_dir) if repo_dir is not None else DEFAULT_REPO_DIR
    _prepare_repo_imports(repo_dir)

    from model.DefocusNet_backbone_defocus_gt import DefocusNet

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = DefocusNet(num_classes=1, backbone=backbone)

    if weights_path is None:
        weights_path = repo_dir / "weights" / "defocus_gt_legacy_xception_Deepfakes.pth"
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find weights file: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict):
        state_dict = (
            ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt.get("model")
            or ckpt
        )
    else:
        state_dict = ckpt

    clean_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "", 1) if k.startswith("module.") else k
        clean_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=strict)
    print(f"Loaded Defocus model weights: {weights_path}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model, device


# ---------- prediction ----------
@torch.no_grad()
def predict_dataloader_defocus(
    model,
    dataloader,
    device,
    checkpoint_file=None,
    checkpoint_every=100,
    threshold=0.5,
    invert_labels=False,
    suppress_model_prints=True,
    desc="",
):
    """
    Official training uses BCEWithLogitsLoss and sigmoid(output) > 0.5.
    By default, we treat sigmoid(output) as fake_score and pred_label=1 when score >= threshold.

    If a smoke test shows labels are flipped, set invert_labels=True.
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

        # The official model prints timers and tensor min/max every forward.
        # Suppress it so tqdm and notebooks stay usable.
        if suppress_model_prints:
            with contextlib.redirect_stdout(io.StringIO()):
                _, logits = model(images)
        else:
            _, logits = model(images)

        fake_scores = torch.sigmoid(logits).detach().float().cpu().view(-1).tolist()

        for sample_id, split, true_label, fake_score in zip(
            batch["ids"], batch["splits"], batch["true_labels"], fake_scores
        ):
            pred_label = 1 if fake_score >= threshold else 0
            if invert_labels:
                pred_label = 1 - pred_label

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


def _build_result(rows, model_name, dataset_name, dataset_root, threshold, invert_labels):
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
def evaluate_dataset_defocus_fast(
    dataset_name,
    dataset_root,
    model,
    device,
    model_name="Defocus_GT_Xception",
    batch_size=16,
    checkpoint_every=100,
    num_workers=0,
    max_batches=None,
    threshold=0.5,
    invert_labels=False,
    image_size=299,
    suppress_model_prints=True,
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

    transform = get_defocus_transform(image_size=image_size)
    dataset = ImageEvalDatasetDefocus(remaining, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_defocus,
        persistent_workers=(num_workers > 0),
    )

    tail_buffer = predict_dataloader_defocus(
        model=model,
        dataloader=dataloader,
        device=device,
        checkpoint_file=checkpoint_file,
        checkpoint_every=checkpoint_every,
        threshold=threshold,
        invert_labels=invert_labels,
        suppress_model_prints=suppress_model_prints,
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


def evaluate_all_datasets_defocus_fast(
    model,
    device,
    model_name="Defocus_GT_Xception",
    batch_size=16,
    checkpoint_every=100,
    num_workers=0,
    max_batches=None,
    threshold=0.5,
    invert_labels=False,
    image_size=299,
    suppress_model_prints=True,
):
    dataset_roots = get_dataset_roots()
    all_results = {}

    items = list(dataset_roots.items())
    total = len(items)

    for idx, (dataset_name, dataset_root) in enumerate(items, start=1):
        prefix = f"[{idx}/{total}] "
        print(f"\n{prefix}Running dataset: {dataset_name}")
        all_results[dataset_name] = evaluate_dataset_defocus_fast(
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
            suppress_model_prints=suppress_model_prints,
            progress_prefix=prefix,
        )

    return all_results


if __name__ == "__main__":
    model, device = load_defocus_model(
        repo_dir="Defocus-Deepfake-Detection",
        weights_path="Defocus-Deepfake-Detection/weights/defocus_gt_legacy_xception_Deepfakes.pth",
        backbone="legacy_xception",
    )

    results = evaluate_all_datasets_defocus_fast(
        model=model,
        device=device,
        model_name="Defocus_GT_Xception_Deepfakes",
        batch_size=8,
        checkpoint_every=50,
        num_workers=0,
        max_batches=3,
        threshold=0.5,
        invert_labels=False,
    )

    print(json.dumps(results, indent=2))
