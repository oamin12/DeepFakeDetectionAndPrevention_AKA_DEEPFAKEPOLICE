from pathlib import Path

# 0 = real, 1 = fake
LABEL_FROM_FOLDER = {
    "real": 0,
    "fake": 1,
    "deepfake": 1,
    "ai_images": 1,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_dataset_roots():
    base = Path.cwd() / "Datasets"
    return {
        "20K_real_and_deepfake_images": base / "20K_real_and_deepfake_images",
        "DeepDetect-2025": base / "DeepDetect-2025" / "ddata",          # e7tmal yb2a msh bye2raha sa7
        "Deepfake-vs-Real-v2": base / "Deepfake-vs-Real-v2",
        "gravex200k": base / "gravex200k" / "my_real_vs_ai_dataset" / "my_real_vs_ai_dataset",
    }


def collect_images(dataset_root):
    dataset_root = Path(dataset_root)
    samples = []

    for class_name, label in LABEL_FROM_FOLDER.items():
        for folder in dataset_root.rglob(class_name):
            if ".cache" in folder.parts:
                continue
            if not folder.is_dir():
                continue

            split = "all"
            for p in folder.parts:
                if p.lower() in {"train", "test", "val", "valid", "validation"}:
                    split = p.lower()

            for img_path in folder.rglob("*"):
                if (
                    img_path.is_file()
                    and img_path.suffix.lower() in IMAGE_EXTS
                    and ".cache" not in img_path.parts
                ):
                    samples.append({
                        "id": str(img_path.relative_to(dataset_root)).replace("\\", "/"),
                        "path": str(img_path),
                        "true_label": label,
                        "split": split,
                    })

    samples.sort(key=lambda x: x["id"])
    return samples


def compute_metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def map_pred_label(raw_label, label_map=None):
    if label_map is not None:
        if raw_label in label_map:
            return label_map[raw_label]
        if isinstance(raw_label, str):
            raw = raw_label.lower().strip()
            for k, v in label_map.items():
                if isinstance(k, str) and k.lower().strip() == raw:
                    return v

    if isinstance(raw_label, str):
        raw = raw_label.lower().strip()
        if "real" in raw or "human" in raw or "authentic" in raw:
            return 0
        if "fake" in raw or "deepfake" in raw or "ai" in raw or "synthetic" in raw:
            return 1

    if raw_label in [0, 1]:
        return raw_label

    raise ValueError(f"Could not map predicted label: {raw_label}")
