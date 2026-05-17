from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


DEFAULT_DATASETS = [
    "20K_real_and_deepfake_images",
    "DeepDetect-2025",
    "Deepfake-vs-Real-v2",
    "gravex200k",
    "nuriachandra_Deepfake-Eval-2024",
]

METRIC_ALIASES = {
    "accuracy": ["accuracy", "acc"],
    "precision": ["precision", "prec"],
    "recall": ["recall", "rec"],
    "f1": ["f1", "f1_score", "f1-score", "f1score"],
}


def get_known_dataset_names() -> List[str]:
    """Prefer dataset names from utils.py, fallback to the known project names."""
    try:
        from utils import get_dataset_roots  # type: ignore

        names = list(get_dataset_roots().keys())
        if names:
            return names
    except Exception:
        pass
    return DEFAULT_DATASETS


def parse_model_dataset_from_filename(path: Path, dataset_names: List[str]) -> Tuple[str, str]:
    """
    Parse file naming convention:
        Results/{model_name}_{dataset_name}.json

    Because dataset names contain underscores, match the longest known dataset suffix.
    """
    stem = path.stem
    for dataset_name in sorted(dataset_names, key=len, reverse=True):
        suffix = "_" + dataset_name
        if stem.endswith(suffix):
            model_name = stem[: -len(suffix)]
            return model_name, dataset_name
    return stem, "UNKNOWN_DATASET"


def normalize_metric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def get_metric(metrics: Dict[str, Any], canonical_name: str) -> Optional[float]:
    aliases = METRIC_ALIASES[canonical_name]

    # Exact alias match first
    for alias in aliases:
        if alias in metrics:
            return normalize_metric_value(metrics[alias])

    # Case-insensitive fallback
    lowered = {str(k).lower(): v for k, v in metrics.items()}
    for alias in aliases:
        if alias.lower() in lowered:
            return normalize_metric_value(lowered[alias.lower()])

    return None


def fmt_metric(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def load_result_json(path: Path, dataset_names: List[str]) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Skipping invalid JSON: {path} ({exc})")
        return None

    parsed_model, parsed_dataset = parse_model_dataset_from_filename(path, dataset_names)

    model_name = data.get("model_name") or parsed_model
    dataset_name = data.get("dataset_name") or parsed_dataset

    metrics = data.get("metrics", {}) or {}
    counts = data.get("counts", {}) or {}

    row = {
        "file": str(path),
        "model_name": str(model_name),
        "dataset_name": str(dataset_name),
        "num_samples": data.get("num_samples"),
        "threshold": data.get("threshold"),
        "invert_labels": data.get("invert_labels"),
        "accuracy": get_metric(metrics, "accuracy"),
        "precision": get_metric(metrics, "precision"),
        "recall": get_metric(metrics, "recall"),
        "f1": get_metric(metrics, "f1"),
        "TP": counts.get("TP"),
        "TN": counts.get("TN"),
        "FP": counts.get("FP"),
        "FN": counts.get("FN"),
    }
    return row


def collect_results(results_dir: Path, dataset_names: List[str]) -> List[Dict[str, Any]]:
    files = sorted(results_dir.glob("*.json"))
    rows = []
    for path in files:
        # Skip hidden/system files if any
        if path.name.startswith("."):
            continue
        row = load_result_json(path, dataset_names)
        if row is not None:
            rows.append(row)
    return rows


def dedupe_latest(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """If the same model/dataset appears more than once, keep the latest modified file."""
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["model_name"], row["dataset_name"])
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        try:
            new_mtime = Path(row["file"]).stat().st_mtime
            old_mtime = Path(current["file"]).stat().st_mtime
            if new_mtime >= old_mtime:
                best[key] = row
        except Exception:
            best[key] = row
    return list(best.values())


def metric_cell(row: Dict[str, Any], decimals: int, include_counts: bool = False) -> str:
    parts = [
        f"Acc: {fmt_metric(row['accuracy'], decimals)}",
        f"Prec: {fmt_metric(row['precision'], decimals)}",
        f"Rec: {fmt_metric(row['recall'], decimals)}",
        f"F1: {fmt_metric(row['f1'], decimals)}",
    ]
    if include_counts:
        parts.append(
            f"TP/TN/FP/FN: {row.get('TP', '-')}/{row.get('TN', '-')}/{row.get('FP', '-')}/{row.get('FN', '-')}"
        )
    return "<br>".join(parts)


def build_wide_table(
    rows: List[Dict[str, Any]],
    dataset_names: List[str],
    decimals: int = 4,
    include_counts: bool = False,
) -> Tuple[List[str], List[List[str]]]:
    model_names = sorted({r["model_name"] for r in rows})
    dataset_names_present = [
        d for d in dataset_names if any(r["dataset_name"] == d for r in rows)
    ]
    # Include unknown/new datasets at the end
    extra_datasets = sorted(
        {r["dataset_name"] for r in rows if r["dataset_name"] not in dataset_names_present}
    )
    dataset_order = dataset_names_present + extra_datasets

    lookup = {(r["model_name"], r["dataset_name"]): r for r in rows}

    headers = ["Model"] + dataset_order
    table = []
    for model_name in model_names:
        line = [model_name]
        for dataset_name in dataset_order:
            row = lookup.get((model_name, dataset_name))
            if row is None:
                line.append("-")
            else:
                line.append(metric_cell(row, decimals=decimals, include_counts=include_counts))
        table.append(line)
    return headers, table


def markdown_table(headers: List[str], table: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in table:
        safe = [str(x).replace("\n", "<br>") for x in row]
        lines.append("| " + " | ".join(safe) + " |")
    return "\n".join(lines)


def write_csv(path: Path, headers: List[str], table: List[List[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in table:
            # Excel/CSV should use real line breaks, not HTML <br>
            writer.writerow([str(x).replace("<br>", "\n") for x in row])


def write_flat_csv(path: Path, rows: List[Dict[str, Any]], decimals: int) -> None:
    headers = [
        "model_name",
        "dataset_name",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "num_samples",
        "threshold",
        "invert_labels",
        "TP",
        "TN",
        "FP",
        "FN",
        "file",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["model_name"], r["dataset_name"])):
            out = dict(row)
            for m in ["accuracy", "precision", "recall", "f1"]:
                out[m] = fmt_metric(out.get(m), decimals)
            writer.writerow({h: out.get(h, "") for h in headers})


def try_write_xlsx(path: Path, headers: List[str], table: List[List[str]]) -> bool:
    """
    Optional convenience: writes XLSX if pandas/openpyxl are installed.
    The MD and CSV outputs do not depend on this.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False

    try:
        df = pd.DataFrame(table, columns=headers)
        # Convert HTML line breaks to Excel line breaks
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace("<br>", "\n", regex=False)
        df.to_excel(path, index=False)
        return True
    except Exception as exc:
        print(f"Could not write XLSX: {exc}")
        return False


def create_results_summary(
    results_dir: str | Path = "Results",
    output_dir: str | Path | None = None,
    output_name: str = "model_results_summary",
    decimals: int = 4,
    include_counts: bool = False,
    write_xlsx: bool = True,
    print_markdown: bool = True,
) -> Dict[str, Any]:
    results_dir = Path(results_dir)
    if output_dir is None:
        output_dir = results_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    dataset_names = get_known_dataset_names()
    rows = collect_results(results_dir, dataset_names)
    rows = dedupe_latest(rows)

    if not rows:
        raise RuntimeError(f"No result JSON files found in: {results_dir}")

    headers, table = build_wide_table(
        rows,
        dataset_names=dataset_names,
        decimals=decimals,
        include_counts=include_counts,
    )

    md = markdown_table(headers, table)

    md_path = output_dir / f"{output_name}.md"
    csv_path = output_dir / f"{output_name}.csv"
    flat_csv_path = output_dir / f"{output_name}_flat.csv"
    xlsx_path = output_dir / f"{output_name}.xlsx"

    md_path.write_text(md + "\n", encoding="utf-8")
    write_csv(csv_path, headers, table)
    write_flat_csv(flat_csv_path, rows, decimals=decimals)

    # xlsx_written = False
    # if write_xlsx:
    #     xlsx_written = try_write_xlsx(xlsx_path, headers, table)

    if print_markdown:
        print(md)

    print("\nSaved files:")
    print(f"- Markdown: {md_path}")
    print(f"- CSV wide table: {csv_path}")
    print(f"- CSV flat table: {flat_csv_path}")
    # if write_xlsx:
    #     if xlsx_written:
    #         print(f"- Excel: {xlsx_path}")
    #     else:
    #         print("- Excel: skipped because pandas/openpyxl is not available")

    return {
        "markdown_path": str(md_path),
        "csv_path": str(csv_path),
        "flat_csv_path": str(flat_csv_path),
        # "xlsx_path": str(xlsx_path) if xlsx_written else None,
        "num_result_files": len(rows),
        "models": sorted({r["model_name"] for r in rows}),
        "datasets": headers[1:],
        "markdown": md,
    }
