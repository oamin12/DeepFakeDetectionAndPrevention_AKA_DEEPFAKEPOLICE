from pathlib import Path
import json
from PIL import Image
from tqdm.auto import tqdm


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
        "DeepDetect-2025": base / "DeepDetect-2025" / "ddata",
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


def predict_batch(image_paths, pipeline_obj, label_map=None):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    outputs = pipeline_obj(images)

    preds = []
    for out in outputs:
        # if model returns list of candidates, take highest score
        if isinstance(out, list):
            out = max(out, key=lambda x: x.get("score", 0.0))

        pred_label = map_pred_label(out["label"], label_map=label_map)
        preds.append({
            "pred_label": pred_label,
            "score": float(out.get("score", 0.0)),
        })
    return preds


def evaluate_dataset(
    dataset_name,
    dataset_root,
    pipeline_obj,
    model_name="model",
    batch_size=16,
    label_map=None,
    progress_prefix="",
    checkpoint_every=50,
    max_batches=None,
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
                    row = json.loads(line)
                    done_ids.add(row["id"])
        print(f"{progress_prefix}Resuming from checkpoint: {len(done_ids)} images already processed")

    remaining = [x for x in samples if x["id"] not in done_ids]
    print(f"{progress_prefix}Remaining images to process: {len(remaining)}")

    buffer = []

    with open(checkpoint_file, "a", encoding="utf-8") as f:
        batch_starts = list(range(0, len(remaining), batch_size))
        if max_batches is not None:
            batch_starts = batch_starts[:max_batches]

        for batch_idx, i in enumerate(
            tqdm(batch_starts, desc=f"{progress_prefix}{dataset_name}"),
            start=1
        ):
            batch = remaining[i:i + batch_size]
            image_paths = [x["path"] for x in batch]
            preds = predict_batch(image_paths, pipeline_obj, label_map=label_map)

            for sample, pred in zip(batch, preds):
                row = {
                    "id": sample["id"],
                    "split": sample["split"],
                    "true_label": sample["true_label"],
                    "pred_label": pred["pred_label"],
                    "score": pred["score"],
                }
                buffer.append(row)

            if batch_idx % checkpoint_every == 0:
                for row in buffer:
                    f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{progress_prefix}Checkpoint saved: {len(buffer)} new predictions -> {checkpoint_file}")
                buffer = []

        if buffer:
            for row in buffer:
                f.write(json.dumps(row) + "\n")
            f.flush()

    # Load all predictions from checkpoint
    rows = []
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    tp = tn = fp = fn = 0
    # misclassified = []

    for row in rows:
        y_true = row["true_label"]
        y_pred = row["pred_label"]

        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
            # misclassified.append(row)
        elif y_true == 1 and y_pred == 0:
            fn += 1
            # misclassified.append(row)

    result = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "num_samples": len(rows),
        "counts": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        },
        "metrics": compute_metrics(tp, tn, fp, fn),
        "misclassified_ids": {
            "FP": [x["id"] for x in rows if x["true_label"] == 0 and x["pred_label"] == 1],
            "FN": [x["id"] for x in rows if x["true_label"] == 1 and x["pred_label"] == 0],
        }
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def evaluate_all_datasets(
    pipeline_obj,
    model_name="model",
    batch_size=16,
    label_map=None,
    checkpoint_every=50,
    max_batches=None,
):
    dataset_roots = get_dataset_roots()
    all_results = {}

    items = list(dataset_roots.items())
    total = len(items)

    for idx, (dataset_name, dataset_root) in enumerate(items, start=1):
        prefix = f"[{idx}/{total}] "
        print(f"\n{prefix}Running dataset: {dataset_name}")
        all_results[dataset_name] = evaluate_dataset(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            pipeline_obj=pipeline_obj,
            model_name=model_name,
            batch_size=batch_size,
            label_map=label_map,
            progress_prefix=prefix,
            checkpoint_every=checkpoint_every,
            max_batches=max_batches,
        )

    return all_results