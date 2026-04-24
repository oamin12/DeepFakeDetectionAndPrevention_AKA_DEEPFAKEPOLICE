from pathlib import Path
import json
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from utils import get_dataset_roots, collect_images, compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_DIR = PROJECT_ROOT / "M2F2_Det"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from sequence.models.M2F2_Det.models.model import M2F2Det
from sequence.runjobs_utils import torch_load_model
from dataset import get_image_transformation_from_cfg

def get_val_transformation_cfg():
    return {
        'post': {
            'blur': {
                'prob': 0.0,
                'sig': [0.0, 3.0]
            },
            'jpeg': {
                'prob': 0.0,
                'method': ['cv2', 'pil'],
                'qual': [30, 100]
            },
            'noise': {
                'prob': 0.0,
                'var': [0.01]
            }
        },
        'flip': False,
    }


def get_m2f2_transform():
    cfg = get_val_transformation_cfg()
    return get_image_transformation_from_cfg(cfg)


# ---------- dataset ----------
class ImageEvalDatasetM2F2(Dataset):
    def __init__(self, samples, transform, image_size=336):
        self.samples = samples
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")

        image = self.transform(image)

        # Ensure all tensors have the same shape
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return {
            "image": image,
            "id": sample["id"],
            "split": sample["split"],
            "true_label": sample["true_label"],
        }


def collate_m2f2(batch):
    images = torch.stack([x["image"] for x in batch], dim=0)
    ids = [x["id"] for x in batch]
    splits = [x["split"] for x in batch]
    true_labels = [x["true_label"] for x in batch]
    return {
        "images": images,
        "ids": ids,
        "splits": splits,
        "true_labels": true_labels,
    }


# ---------- model loading ----------
def load_m2f2_stage1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = M2F2Det(
        clip_text_encoder_name="openai/clip-vit-large-patch14-336",
        clip_vision_encoder_name="openai/clip-vit-large-patch14-336",
        deepfake_encoder_name="efficientnet_b4",
        hidden_size=1792,
    )

    vision_tower_path = REPO_DIR / "utils" / "weights" / "vision_tower.pth"
    ckpt_path = REPO_DIR / "checkpoints" / "stage_1" / "current_model_180.pth"

    llava_vision_tower = torch.load(vision_tower_path, map_location="cpu", weights_only=True)
    vision_tower_dict = {
        k.replace("vision_tower.", ""): v
        for k, v in llava_vision_tower.items()
    }

    if model.clip_vision_encoder is not None:
        model.clip_vision_encoder.model.load_state_dict(vision_tower_dict, strict=True)
        print("Loaded vision tower.")

    model = torch.nn.DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.module.assign_lr_dict_list(lr=1e-3), weight_decay=1e-6)
    _ = torch_load_model(model, optimizer, str(ckpt_path))

    model.eval()
    return model, device


# ---------- prediction ----------
@torch.no_grad()
def predict_dataloader_m2f2(
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
        images = batch["images"].float().to(device, non_blocking=True)

        out = model(images, return_dict=True)
        logits = out["pred"]
        probs = F.softmax(logits, dim=-1)

        # Repo inference uses class 0 probability as the detection score.
        real_scores = probs[:, 0].tolist()

        for sample_id, split, true_label, real_score in zip(
            batch["ids"], batch["splits"], batch["true_labels"], real_scores
        ):
            pred_label = 0 if real_score >= threshold else 1

            buffer.append({
                "id": sample_id,
                "split": split,
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "real_score": float(real_score),
            })

        if checkpoint_file is not None and batch_idx % checkpoint_every == 0:
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                for row in buffer:
                    f.write(json.dumps(row) + "\n")
                f.flush()
            buffer = []

    return buffer


# ---------- evaluation ----------
def evaluate_dataset_m2f2_fast(
    dataset_name,
    dataset_root,
    model,
    device,
    transform,
    model_name="M2F2_stage1",
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

    dataset = ImageEvalDatasetM2F2(remaining, transform=transform, image_size=336)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_m2f2,
        persistent_workers=(num_workers > 0),
    )

    tail_buffer = predict_dataloader_m2f2(
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


def evaluate_all_datasets_m2f2_fast(
    model,
    device,
    transform,
    model_name="M2F2_stage1",
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
        all_results[dataset_name] = evaluate_dataset_m2f2_fast(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            model=model,
            device=device,
            transform=transform,
            model_name=model_name,
            batch_size=batch_size,
            checkpoint_every=checkpoint_every,
            num_workers=num_workers,
            max_batches=max_batches,
            threshold=threshold,
            progress_prefix=prefix,
        )

    return all_results