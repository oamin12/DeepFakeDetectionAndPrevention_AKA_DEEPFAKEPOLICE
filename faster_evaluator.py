from pathlib import Path
import json
from PIL import Image
from tqdm.notebook import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_dataset_roots, collect_images, compute_metrics, map_pred_label



class ImageEvalDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        return {
            "image": image,
            "id": sample["id"],
            "split": sample["split"],
            "true_label": sample["true_label"],
        }


class CollateFn:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = [x["image"] for x in batch]
        ids = [x["id"] for x in batch]
        splits = [x["split"] for x in batch]
        true_labels = [x["true_label"] for x in batch]

        inputs = self.processor(images=images, return_tensors="pt")

        return {
            "inputs": inputs,
            "ids": ids,
            "splits": splits,
            "true_labels": true_labels,
        }


def predict_dataloader(model, dataloader, device, label_map=None, checkpoint_file=None, checkpoint_every=100, desc="",):
    model.eval()
    buffer = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc), start=1):
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch["inputs"].items()}
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1).tolist()
            scores = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()

            id2label = getattr(model.config, "id2label", None)

            for sample_id, split, true_label, pred_id, score in zip(
                batch["ids"], batch["splits"], batch["true_labels"], pred_ids, scores
            ):
                raw_label = id2label[pred_id] if id2label else pred_id
                pred_label = map_pred_label(raw_label, label_map=label_map)

                buffer.append({
                    "id": sample_id,
                    "split": split,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "score": float(score),
                })

            if checkpoint_file is not None and batch_idx % checkpoint_every == 0:
                with open(checkpoint_file, "a", encoding="utf-8") as f:
                    for row in buffer:
                        f.write(json.dumps(row) + "\n")
                    f.flush()
                buffer = []

    return buffer


def evaluate_dataset_fast(
    dataset_name,
    dataset_root,
    model,
    processor,
    model_name="model",
    batch_size=16,
    label_map=None,
    progress_prefix="",
    checkpoint_every=100,
    max_batches=None,
    num_workers=4,
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

    dataset = ImageEvalDataset(remaining)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=CollateFn(processor),
        persistent_workers=(num_workers > 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tail_buffer = predict_dataloader(
        model=model,
        dataloader=dataloader,
        device=device,
        label_map=label_map,
        checkpoint_file=checkpoint_file,
        checkpoint_every=checkpoint_every,
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

def evaluate_all_datasets_fast(
    model,
    processor,
    model_name="model",
    batch_size=16,
    label_map=None,
    checkpoint_every=100,
    num_workers=4,
    max_batches=None,
):
    dataset_roots = get_dataset_roots()
    all_results = {}

    items = list(dataset_roots.items())
    total = len(items)

    for idx, (dataset_name, dataset_root) in enumerate(items, start=1):
        prefix = f"[{idx}/{total}] "
        print(f"\n{prefix}Running dataset: {dataset_name}")
        all_results[dataset_name] = evaluate_dataset_fast(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            model=model,
            processor=processor,
            model_name=model_name,
            batch_size=batch_size,
            label_map=label_map,
            checkpoint_every=checkpoint_every,
            num_workers=num_workers,
            max_batches=max_batches,
            progress_prefix=prefix,
        )

    return all_results