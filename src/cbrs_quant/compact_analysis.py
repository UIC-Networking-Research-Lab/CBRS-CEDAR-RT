"""Compatibility entry point for per-model RT timing bars and quality matrices."""
from __future__ import annotations
import csv
from io import StringIO
import json
from pathlib import Path
from .analysis import read_bytes
from .report_plots import model_chart
from .rt_report import cohorts, enrich


def newest_run(root):
    runs = [p for p in root.iterdir() if p.is_dir() and any(p.glob("*/model.json"))] if root.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"No compact experiment under {root}")
    def created(path):
        meta = json.loads(read_bytes(next(path.glob("*/model.json")), path.parent))
        return meta["experiment"]["created_at"], path.name
    return max(runs, key=created)


def load_comparison(path, scratch_parent):
    return list(csv.DictReader(StringIO(read_bytes(path, scratch_parent).decode("utf-8-sig"))))


def plot(run: Path, output: Path, formats=("png", "pdf"), dpi=180, baseline="torch_cuda_fp32", scope="image_to_detection"):
    files = []
    for path in sorted(run.glob("*/model.json")):
        meta = json.loads(read_bytes(path, run.parent))
        values = load_comparison(path.parent / "results.csv", run.parent)
        rows = [r for r in enrich(meta, values, baseline) if r["scope"] == scope]
        groups = cohorts(rows)
        for signature, group in groups.items():
            suffix = "_"+signature[0][:8]+"_"+signature[1][:8] if len(groups)>1 else ""
            files.extend(model_chart(meta["model_id"], group, output, scope, formats, dpi, suffix))
        print(f"  Plot: {meta['model_id']}", flush=True)
    return files


def plot_cohort(meta, rows, output, formats, dpi, baseline, scope, suffix):
    return model_chart(meta["model_id"], rows, output, scope, formats, dpi, suffix)
