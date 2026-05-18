from pathlib import Path
import json
import random
import textwrap
from math import ceil

import matplotlib.pyplot as plt
from PIL import Image, ImageFile

from utils import collect_images

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_result_jsons(results_dir="Results"):
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    items = []
    for path in sorted(results_dir.glob("*.json")):
        # Skip non-model summary JSONs if any
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue
        if "misclassified_ids" not in data:
            continue
        if "metrics" not in data and "counts" not in data:
            continue

        items.append((path, data))
    return items


def list_result_models(results_dir="Results"):
    """
    Print available model/dataset combinations found in Results/*.json.
    """
    rows = []
    for path, data in _load_result_jsons(results_dir):
        rows.append({
            "model_name": data.get("model_name", "UNKNOWN"),
            "dataset_name": data.get("dataset_name", "UNKNOWN"),
            "num_samples": data.get("num_samples", "?"),
            "file": str(path),
        })

    if not rows:
        print(f"No result JSON files found in {results_dir}")
        return []

    # print("Available result files:")
    # for r in rows:
    #     print(f"- model={r['model_name']} | dataset={r['dataset_name']} | samples={r['num_samples']} | file={r['file']}")
    return rows


def _matches(value, query, match_mode="exact"):
    if query is None:
        return True
    value = str(value)
    query = str(query)
    if match_mode == "exact":
        return value == query
    if match_mode == "contains":
        return query.lower() in value.lower()
    raise ValueError("match_mode must be 'exact' or 'contains'")


def _resolve_image_paths(dataset_root, image_ids):
    """
    Resolve result JSON image IDs back to actual image file paths.
    Uses collect_images() first because it follows the same ID convention as the evaluators.
    """
    dataset_root = Path(dataset_root)
    samples = collect_images(dataset_root)
    id_to_sample = {s["id"]: s for s in samples}

    resolved = []
    missing = []
    for image_id in image_ids:
        if image_id in id_to_sample:
            sample = id_to_sample[image_id]
            resolved.append({
                "id": image_id,
                "path": Path(sample["path"]),
                "split": sample.get("split"),
                "true_label": int(sample.get("true_label", -1)),
            })
            continue

        # Fallback if the file exists directly by relative ID
        fallback_path = dataset_root / Path(str(image_id).replace("/", str(Path('/').name or '/')))
        # On Windows/Unix Path(dataset_root) / image_id with forward slashes usually works:
        fallback_path = dataset_root / str(image_id)
        if fallback_path.exists():
            resolved.append({
                "id": image_id,
                "path": fallback_path,
                "split": None,
                "true_label": -1,
            })
        else:
            missing.append(image_id)

    return resolved, missing


def _plot_group(
    rows,
    title,
    num_images,
    seed=None,
    cols=4,
    image_size=3.2,
    title_fontsize=7,
    title_pad=4,
    hspace=0.28,
):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    if not rows:
        print(f"No images found for {title}")
        return []

    selected = rng.sample(rows, k=min(num_images, len(rows)))

    cols = max(1, int(cols))
    rows_count = ceil(len(selected) / cols)

    fig, axes = plt.subplots(
        rows_count,
        cols,
        figsize=(cols * image_size, rows_count * (image_size + 0.35)),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for ax in axes_flat:
        ax.axis("off")

    for idx, item in enumerate(selected):
        ax = axes_flat[idx]

        img = Image.open(item["path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        pred_text = "pred=fake" if item["kind"] == "FP" else "pred=real"
        true_text = "true=real" if item["kind"] == "FP" else "true=fake"
        small_id = textwrap.shorten(item["id"], width=34, placeholder="...")

        ax.set_title(
            f"{item['kind']} | {item['dataset_name']}\n{true_text}, {pred_text}\n{small_id}",
            fontsize=title_fontsize,
            pad=title_pad,
        )

    fig.suptitle(title, fontsize=14, y=0.995)

    fig.subplots_adjust(
        top=0.96,
        bottom=0.02,
        left=0.02,
        right=0.98,
        wspace=0.08,
        hspace=hspace,
    )

    plt.show()

    return selected

def show_random_misclassified_images(
    model_name,
    dataset_name=None,
    num_images=8,
    results_dir="Results",
    seed=None,
    match_mode="exact",
    cols=4,
    show_fp=True,
    show_fn=True,
    hspace=0.28,
):
    """
    Randomly visualize misclassified images for a model.

    Parameters
    ----------
    model_name : str
        Model name to search for. Matches the `model_name` field inside result JSON files.
    dataset_name : str | None
        Optional dataset name. If None, images are sampled randomly from all datasets for this model.
    num_images : int
        Maximum number of images to show PER group: FP and FN.
        Example: num_images=5 can show up to 5 FP + 5 FN.
    results_dir : str
        Folder containing evaluator result JSON files.
    seed : int | None
        Random seed for reproducible sampling.
    match_mode : {'exact', 'contains'}
        Use 'exact' for exact model/dataset names, or 'contains' for partial matching.
    cols : int
        Number of columns in each displayed figure.
    show_fp : bool
        Show false positives: true real, predicted fake.
    show_fn : bool
        Show false negatives: true fake, predicted real.

    Returns
    -------
    dict with selected FP/FN rows and matched result files.
    """
    if seed is not None:
        random.seed(seed)

    all_results = _load_result_jsons(results_dir)

    matched = []
    for path, data in all_results:
        json_model_name = data.get("model_name", path.stem)
        json_dataset_name = data.get("dataset_name", "")

        if not _matches(json_model_name, model_name, match_mode=match_mode):
            continue
        if dataset_name is not None and not _matches(json_dataset_name, dataset_name, match_mode=match_mode):
            continue

        matched.append((path, data))

    if not matched:
        print("No matching result JSON files found.")
        print(f"Requested model_name={model_name!r}, dataset_name={dataset_name!r}, match_mode={match_mode!r}")
        print()
        list_result_models(results_dir)
        return {"FP": [], "FN": [], "matched_files": []}

    print("Matched result files:")
    for path, data in matched:
        metrics = data.get("metrics", {})
        acc = metrics.get("accuracy", metrics.get("acc", None))
        print(f"- {path} | model={data.get('model_name')} | dataset={data.get('dataset_name')} | accuracy={acc}")
    print()

    fp_rows = []
    fn_rows = []
    missing_total = []

    for path, data in matched:
        dataset_root = data.get("dataset_root")
        current_dataset_name = data.get("dataset_name", "UNKNOWN")
        current_model_name = data.get("model_name", model_name)

        if not dataset_root:
            print(f"Skipping {path}: dataset_root missing in JSON")
            continue

        mis = data.get("misclassified_ids", {})
        fp_ids = mis.get("FP", []) or []
        fn_ids = mis.get("FN", []) or []

        resolved_fp, missing_fp = _resolve_image_paths(dataset_root, fp_ids)
        resolved_fn, missing_fn = _resolve_image_paths(dataset_root, fn_ids)

        for r in resolved_fp:
            r.update({
                "kind": "FP",
                "dataset_name": current_dataset_name,
                "model_name": current_model_name,
                "result_file": str(path),
            })
            fp_rows.append(r)

        for r in resolved_fn:
            r.update({
                "kind": "FN",
                "dataset_name": current_dataset_name,
                "model_name": current_model_name,
                "result_file": str(path),
            })
            fn_rows.append(r)

        missing_total.extend([(current_dataset_name, "FP", x) for x in missing_fp])
        missing_total.extend([(current_dataset_name, "FN", x) for x in missing_fn])

    print(f"Total FP available: {len(fp_rows)}")
    print(f"Total FN available: {len(fn_rows)}")
    if missing_total:
        print(f"Warning: {len(missing_total)} misclassified IDs could not be resolved to files.")
        print("First missing examples:")
        for item in missing_total[:10]:
            print(" ", item)
    print()

    selected_fp = []
    selected_fn = []

    if show_fp:
        selected_fp = _plot_group(
            fp_rows,
            title=f"False Positives for {model_name} (true real, predicted fake)",
            num_images=num_images,
            seed=seed,
            cols=cols,
            hspace=hspace,
        )

    if show_fn:
        # Use a shifted seed so FP and FN sampling are independent but reproducible.
        fn_seed = None if seed is None else seed + 999
        selected_fn = _plot_group(
            fn_rows,
            title=f"False Negatives for {model_name} (true fake, predicted real)",
            num_images=num_images,
            seed=fn_seed,
            cols=cols,
        )

    # print("Selected FP images:")
    # for x in selected_fp:
    #     print(f"- [{x['dataset_name']}] {x['id']} -> {x['path']}")

    # print("\nSelected FN images:")
    # for x in selected_fn:
    #     print(f"- [{x['dataset_name']}] {x['id']} -> {x['path']}")

    return {
        "FP": selected_fp,
        "FN": selected_fn,
        "matched_files": [str(path) for path, _ in matched],
        "missing": missing_total,
    }
