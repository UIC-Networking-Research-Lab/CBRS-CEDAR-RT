"""Timing bars and quality matrices for legacy JSON experiment reports."""
from __future__ import annotations
from collections import defaultdict
from .config import MODES
from .report_plots import COLORS, LABELS, coverage_chart, model_chart, model_label

# Retained for older imports; quality is displayed in matrices.
STYLE = {mode: (COLORS[mode], "s") for mode in MODES}


def render(snapshot, quality, timings, output, baseline, formats=("png", "pdf"), dpi=180, progress=print):
    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    figures = []
    def record(title, files):
        if files:
            figures.append({"title": title, "files": [p.relative_to(output).as_posix() for p in files]})
            progress(f"  Figure {len(figures)}: {title}", flush=True)
    record("Experiment coverage", coverage_chart(snapshot["routes"], destination, formats, dpi))
    quality_lookup = {(r["model"], r["batch"], r["mode"], r.get("quality_signature")): r for r in quality}
    groups = defaultdict(list)
    for original in timings:
        row = dict(original)
        q = quality_lookup.get((row["model"], row["batch"], row["mode"], row.get("quality_signature")), {})
        row["inference_time_per_frame_s"] = 1 / float(row["images_per_second_mean"])
        fs = [q.get("lte_f1"), q.get("radar_f1")]
        row["overlap_f1"] = q.get("overlap_f1")
        row["map50"] = q.get("map50")
        groups[(row["model"], row["scope"], row.get("timing_signature", ""), row.get("quality_signature", ""))].append(row)
    for (model, scope, signature, qsig), rows in groups.items():
        suffix = "_"+signature[:8]+"_"+qsig[:8]
        record(f"{model_label(model)}: {scope}", model_chart(model, rows, destination, scope, formats, dpi, suffix))
    return figures
