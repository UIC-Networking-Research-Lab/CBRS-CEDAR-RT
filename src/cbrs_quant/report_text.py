"""The complete report and concise README summary from one consolidated store."""
from __future__ import annotations

import json

from .analysis import SCOPES
from .config import MODES, CPU_MODES
from .report_plots import LABELS, SCOPE_LABELS, model_label
from .rt_report import fmt, image_link, observations


def write_reports(bundle, rows, routes, charts, overview_files, output):
    selected_routes = [r for r in routes if r["in_scope"]]
    scope = bundle["report_scope"]
    completed = sum(r["status"] == "passed" for r in selected_routes)
    devices = "GPU" if all(m not in CPU_MODES for m in scope["modes"]) else "GPU/CPU"
    completion = f"**{completed}/{len(selected_routes)} {devices} routes completed: {len(scope['models'])} models, batches {', '.join(map(str,scope['batches']))}, {len(scope['modes'])} deployment policies.**"
    first = bundle["models"][bundle["order"][0]]["meta"]
    environments = [env for item in bundle["models"].values() for env in item["meta"].get("environments", {}).values()]
    env = environments[0] if environments else {}
    packages = env.get("packages", {})
    images = sorted({int(r["evaluated_images"]) for r in rows if r["evaluated_images"] != ""})
    split = ", ".join("validation" if s == "val" else s for s in sorted({r["evaluation_split"] for r in rows}))
    audit = json.loads((output / "data/manifest.json").read_text(encoding="utf-8")).get("audit", {})
    parts = ["# CBRS-CEDAR-RT: complete GPU experiment report", "", completion, "",
        f"Recorded experiment: `{bundle['run_name']}`. All final measurements and earlier recorded attempts are stored together in [data/](data/README.md).", "",
        "## Setup", "",
        "**All experiments in this report run on the PC GPU**, in Conda `torch270_py312_cuda`; no Jetson deployment tests are reported. The `trained_models_mdi/` name identifies the CEDAR checkpoint edition, not the execution platform.", "",
        "The [README environment table](../README.md#installation) lists tested/recommended versions. Compatible general runtime libraries can be used. For raw-I/Q conversion, preserve preprocessing library versions and parameters: even small pixel changes can invalidate pretrained predictions. The recorded RT experiments use the supplied PNGs directly.", "",
        f"- Device: **{env.get('gpu','see metadata')}**, {env.get('platform','see metadata')}.",
        f"- Software: Python {env.get('python','see metadata')}; PyTorch {packages.get('torch','see metadata')}; ONNX Runtime {packages.get('onnxruntime-gpu','see metadata')}; TensorRT {packages.get('tensorrt','see metadata')}.",
        f"- Evaluation: **{split} split, {', '.join(map(str,images))} images**. INT8 calibration uses 512 training images.",
        "- Models: YOLOv8, YOLO11 and YOLO26 at nano, small and extra-large sizes; CEDAR-YOLO-n/s/x with a 20 × 20 feature grid. CEDAR weights are from the `trained_models_mdi/` family.",
        f"- Timing: {first['experiment'].get('protocol',{}).get('rounds',3)} rounds per scope, with separate synchronized latency and throughput loops. Each loop uses the recorded minimum duration and image count; inputs and runtime caches are warmed.",
        f"- Scope: {devices} policies at batches {', '.join(map(str,scope['batches']))}. Original plan entries outside this evaluation remain in the data with `in_scope=False`.", "",
        "| Timing scope | Included work |", "|---|---|",
        "| Model computation | Forward pass, with the input tensor already on the device |",
        "| Detector pipeline | Preprocessing, forward pass and box decoding/NMS from a decoded image |",
        "| Image to detection | PNG reading/decoding and the full detector pipeline; warm file cache |", "",
        "## Metrics", "", "| Metric | Meaning |", "|---|---|",
        "| Inference time per frame (s/frame) ↓ | Inverse of mean throughput across the three rounds. At batch > 1, this is time normalized by throughput, not request latency. |",
        "| Detection F1 ↑ | Mean of LTE and radar box-detection F1, at confidence 0.25 and box IoU 0.5. This differs from the paper's frame-level overlap F1. |",
        "| mAP@0.5 ↑ | Mean LTE/radar box AP at IoU 0.5; confidence-greedy matching and 101 recall points. |",
        "| mAP@0.5:0.95 ↑ | Mean AP over IoU thresholds 0.50 to 0.95; retained in all detailed tables. |",
        "| p50 / p95 / p99 | Per-batch request latency percentiles in milliseconds, averaged across rounds. |", "",
        "Quality is evaluated once for each model/batch/policy. Repeated quality columns across timing scopes refer to that same evaluation.", "",
        "## Results", "", *observations(rows), "",
        "Vertical bars compare inference time per frame. Annotated matrices compare F1 and mAP. Quality colors span 0–1; timing panels use a separate scale for each model.", ""]
    for files in overview_files:
        parts.extend([image_link(files, output), ""])
    parts.extend(["## Completion and data audit", "", "| Batch | Completed GPU routes | Remaining |", "|---:|---:|---:|"])
    for batch in scope["batches"]:
        group = [r for r in selected_routes if int(r["batch"]) == batch]
        n = sum(r["status"] == "passed" for r in group)
        parts.append(f"| {batch} | {n}/{len(group)} | {len(group)-n} |")
    parts.extend(["", f"The final tables contain **{len(rows)} model × batch × policy × timing-scope aggregates**. Each aggregate is checked against its original rounds and protocol signatures.", "",
        f"Earlier attempts are retained in `data/history.jsonl` ({audit.get('history_attempts','see manifest')} distinct archived attempts). They are excluded from the final averages. The original full experiment plan and canceled entries remain traceable in `data/routes.csv` and `data/metadata.json`.", "",
        "## All model results", "", "| Model | Final routes | Detailed results |", "|---|---:|---|"])
    for model in scope["models"]:
        group = [r for r in selected_routes if r["model"] == model]
        parts.append(f"| {model_label(model)} | {sum(r['status']=='passed' for r in group)}/{len(group)} | [Tables and figures](#{model}) |")
    for model in scope["models"]:
        parts.extend(["", f'<a id="{model}"></a>', f"### {model_label(model)}", ""])
        for timing_scope in SCOPES:
            parts.extend([f"#### {SCOPE_LABELS[timing_scope]}", ""])
            for files in charts.get((model,timing_scope), []):
                if timing_scope == "model_compute":
                    parts.extend([image_link(files,output), ""])
                else:
                    figure = next((p for p in files if p.suffix == ".png"),files[0])
                    parts.extend([f"[Comparison figure]({figure.relative_to(output).as_posix()})", ""])
            parts.extend(["| Batch | Policy | Time (s/frame) ↓ | F1 ↑ | LTE F1 | Radar F1 | mAP@0.5 ↑ | mAP@0.5:0.95 ↑ | p50 (ms/batch) | p95 (ms/batch) | p99 (ms/batch) |",
                          "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for row in sorted((r for r in rows if r["model"]==model and r["scope"]==timing_scope),key=lambda r:(int(r["batch"]),MODES.index(r["mode"]))):
                fields = [("inference_time_per_frame_s",6),("macro_f1",4),("lte_f1",4),("radar_f1",4),
                          ("map50",4),("map50_95",4),("p50_ms_mean",3),("p95_ms_mean",3),("p99_ms_mean",3)]
                parts.append(f"| {row['batch']} | {LABELS[row['mode']]} | "+" | ".join(fmt(row.get(key),places) for key,places in fields)+" |")
            parts.append("")
    parts.extend(["## Rebuild or analyze", "",
        "```bash\npython -m pip install -r requirements.txt\npython analyze_results.py\n```", "",
        "The command reads `results/data/`, verifies checksums and regenerates this report, all figures and the root README. It needs no GPU, images, weights or former output directories.", "",
        "For your own analysis, read `results/data/summary.csv` directly, or use:", "",
        '```python\nimport sys\nsys.path.insert(0, "src")\nfrom cbrs_quant.results_store import table\nsummary = table("results", "summary")\nrounds = table("results", "rounds")\n```', "",
        "See [data/README.md](data/README.md) for file schemas, scope filtering and provenance.", ""])
    (output / "REPORT.md").write_text("\n".join(parts),encoding="utf-8")
    overview_scope = overview_files[0][0].stem.split("_b")[0].removeprefix("overview_") if overview_files else "model_compute"
    summary = ["## Results", "", completion, "",
        f"**{env.get('gpu','Recorded GPU')} · batch 1 · {SCOPE_LABELS.get(overview_scope,overview_scope).lower()}.** Quality uses the **{split} split ({', '.join(map(str,images))} images)**.", "",
        "**Inference time per frame (s/frame) ↓ · Detection F1 ↑ · mAP@0.5 ↑**. Time is the inverse of mean throughput; F1 averages LTE and radar box-detection F1.", ""]
    example = [r for r in rows if r['model']=='cbrs_yolo8x' and int(r['batch'])==1 and r['scope']==overview_scope
               and r['mode'] in ('torch_cuda_fp32','ort_trt_fp16','ort_trt_int8')]
    if len(example)==3:
        summary.extend(['**Example: YOLOv8x, same checkpoint and validation images.**', '',
                        '| Policy | Time (s/frame) ↓ | Detection F1 ↑ | mAP@0.5 ↑ |',
                        '|---|---:|---:|---:|'])
        for row in example:
            summary.append(f"| {LABELS[row['mode']]} | {fmt(row['inference_time_per_frame_s'],4)} | {fmt(row['macro_f1'],4)} | {fmt(row['map50'],4)} |")
        summary.append('')
    summary.extend(['<details>', '<summary>All 12 models: inference time, F1 and mAP comparisons</summary>', ''])
    for files in overview_files:
        summary.extend([image_link(files,output), ""])
    summary.extend(['</details>', ''])
    summary.extend(["FP16 and INT8 can reduce model computation time; compare detection quality alongside speed. The full report also includes preprocessing and image decoding.", "",
        "[Complete report](REPORT.md) · [Combined measurements](data/summary.csv) · [Data guide](data/README.md)", ""])
    (output / "SUMMARY.md").write_text("\n".join(summary),encoding="utf-8")
    guide = """# Consolidated experiment data

This is the single data location used by `python analyze_results.py`. Files are combined across all models, batches and policies; no per-model result directories are needed.

| File | Contents |
|---|---|
| `summary.csv` | Final full-precision aggregates: model, batch, policy and timing scope; time per frame, throughput, F1, mAP and latency percentiles. |
| `rounds.csv` | Every original row from the latest compact results, including unrun plan placeholders. Filter `status == passed` and a nonempty `scope` for measurements. |
| `routes.csv` | Original full-plan states plus `in_scope`; use `in_scope == True` for the final GPU evaluation. |
| `metadata.json` | All model metadata, experiment protocols, environments, hashes, calibration records and the explicit final report scope. |
| `dataset.json` | Dataset membership, split labels and hashes; no input images. |
| `history.jsonl` | Earlier recorded attempts, including failures and available partial measurements; these are not extra repetitions of the final evaluation. |
| `manifest.json` | SHA-256 checksums, source audit and analysis-code hashes. |

`inference_time_per_frame_s = 1 / mean(images_per_second)` across the three timing rounds. `macro_f1` is the arithmetic mean of LTE and radar box-detection F1. Quality is evaluated once per route and repeated alongside each timing scope.

The final scope is 12 models, batches 1/4/8 and eight GPU policies. CPU testing is canceled. The original plan also included batches 16/32; those unrun entries remain provenance and are excluded from report completion counts.

Numeric measurements are preserved. Private home-directory prefixes are sanitized. Historical runtime and export paths in metadata describe the original runs; analysis reads only files in this directory. Input data and model weights are external assets.

```python
import csv
from pathlib import Path
with Path("results/data/summary.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
```
"""
    (output / "data/README.md").write_text(guide,encoding="utf-8")
