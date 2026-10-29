"""The complete report and concise README summary from one consolidated store."""
from __future__ import annotations

import json

from .analysis import SCOPES
from .config import MODES, CPU_MODES
from .report_plots import LABELS, SCOPE_LABELS, model_label
from .rt_report import fmt, image_link


def overview_sections(rows, overview_files, output, timing_scope):
    """Explain each visible overview using matched batch-one measurements."""
    selected = [r for r in rows if int(r['batch']) == 1 and r['scope'] == timing_scope]
    lookup = {(r['model'], r['mode']): r for r in selected}
    models = list(dict.fromkeys(r['model'] for r in selected))

    def pairs(mode):
        return [(lookup[(model, 'torch_cuda_fp32')], lookup[(model, mode)])
                for model in models if (model, 'torch_cuda_fp32') in lookup and (model, mode) in lookup]

    speed, quality, maps = [], [], []
    for mode, label in [('ort_trt_fp16', 'ONNX Runtime with TensorRT FP16'),
                        ('trt_fp16', 'Direct TensorRT FP16')]:
        matched = pairs(mode)
        if matched:
            ratios = [float(a['inference_time_per_frame_s']) / float(b['inference_time_per_frame_s'])
                      for a, b in matched]
            speed.append(f"{label} achieves **{min(ratios):.2f}–{max(ratios):.2f}× the PyTorch FP32 throughput** "
                         f"across {len(matched)} models in this timing scope.")
    baseline = lookup.get(('cbrs_yolo8x', 'torch_cuda_fp32'))
    cuda16 = lookup.get(('cbrs_yolo8x', 'ort_cuda_fp16'))
    if baseline and cuda16:
        speed.append(f"For YOLOv8x, ORT CUDA FP16 takes {float(cuda16['inference_time_per_frame_s']):.4f} "
                     f"s/frame, compared with "
                     f"{float(baseline['inference_time_per_frame_s']):.4f} s/frame for PyTorch FP32.")

    fp16 = pairs('ort_trt_fp16') + pairs('trt_fp16')
    f1_deltas = [abs(float(b['overlap_f1']) - float(a['overlap_f1'])) for a, b in fp16
                 if a.get('overlap_f1') is not None and b.get('overlap_f1') is not None]
    if f1_deltas:
        quality.append(f"The TensorRT FP16 configurations track each model's PyTorch FP32 quality: "
                       f"the largest absolute overlap-F1 change is **{max(f1_deltas):.5f}**.")
    if fp16:
        delta = max(abs(float(b['map50']) - float(a['map50'])) for a, b in fp16)
        maps.append(f"Across the TensorRT FP16 configurations, the largest absolute mAP@0.5 change "
                    f"from PyTorch FP32 is **{delta:.5f}**.")
    int8 = pairs('trt_int8')
    if int8:
        overlap_pairs = [(a, b) for a, b in int8
                         if a.get('overlap_f1') is not None and b.get('overlap_f1') is not None]
        if overlap_pairs:
            a, b = max(overlap_pairs, key=lambda pair: abs(float(pair[1]['overlap_f1']) - float(pair[0]['overlap_f1'])))
            delta = abs(float(b['overlap_f1']) - float(a['overlap_f1']))
            if delta:
                quality.append(f"The largest absolute Overlap-F1 change with direct TensorRT INT8 is "
                               f"**{delta:.5f}** on {model_label(a['model'])}: "
                               f"**{float(a['overlap_f1']):.4f} to {float(b['overlap_f1']):.4f}**.")
            else:
                quality.append("Direct TensorRT INT8 matches PyTorch FP32 Overlap F1 for every model in this comparison.")
        a, b = max(int8, key=lambda pair: float(pair[0]['map50']) - float(pair[1]['map50']))
        name = model_label(a['model'])
        maps.append(f"The largest mAP@0.5 reduction under direct TensorRT INT8 occurs on {name}: "
                    f"**{float(a['map50']):.4f} to {float(b['map50']):.4f}**. "
                    "INT8 deployment therefore requires validation for each model and batch configuration.")

    definitions = [
        ('inference_time_per_frame_s', 'Inference speed',
         'Inference time per frame is the inverse of mean throughput across the timing rounds; lower is better. '
         'Each panel uses its own time scale. Figure labels: Torch = PyTorch; CUDA = ONNX Runtime CUDA; ORT TRT = ONNX Runtime TensorRT; TRT = direct TensorRT.', speed),
        ('overlap_f1', 'Overlap F1',
         'Rows are deployment policies and columns are models. Overlap F1 measures the frame-level radar/LTE '
         'frequency-overlap decision at confidence 0.25, using the same rule as MDI; higher is better.', quality),
        ('map50', 'mAP@0.5',
         'The matrix reports mean average precision for LTE and radar boxes at IoU 0.5; higher is better.', maps)]
    sections = []
    for metric, title, caption, findings in definitions:
        files = next((group for group in overview_files if any(p.stem.endswith('_' + metric) for p in group)), [])
        if not files:
            continue
        figure = image_link(files, output).replace('Policy comparison', title + ' across deployment policies')
        sections.extend([f'### {title}', '', figure, '', caption, '', ' '.join(findings), ''])
    return sections


def write_reports(bundle, rows, routes, charts, overview_files, output):
    selected_routes = [r for r in routes if r["in_scope"]]
    scope = bundle["report_scope"]
    completed = sum(r["status"] == "passed" for r in selected_routes)
    devices = "GPU" if all(m not in CPU_MODES for m in scope["modes"]) else "GPU/CPU"
    completion = f"**{completed}/{len(selected_routes)} {devices} configurations completed: {len(scope['models'])} models, batches {', '.join(map(str,scope['batches']))}, {len(scope['modes'])} deployment policies.**"
    first = bundle["models"][bundle["order"][0]]["meta"]
    environments = [env for item in bundle["models"].values() for env in item["meta"].get("environments", {}).values()]
    env = environments[0] if environments else {}
    packages = env.get("packages", {})
    images = sorted({int(r["evaluated_images"]) for r in rows if r["evaluated_images"] != ""})
    split = ", ".join("validation" if s == "val" else s for s in sorted({r["evaluation_split"] for r in rows}))
    audit = json.loads((output / "data/manifest.json").read_text(encoding="utf-8")).get("audit", {})
    quality_note = (
        "Overlap F1 is measured in a quality-only supplementary evaluation of each model/batch/policy "
        "using the same checkpoints and validation split. Original timing and mAP measurements are retained."
        if (output / "data/overlap_evaluations.csv").exists() else
        "Quality is evaluated once for each model/batch/policy.")
    parts = ["# CBRS-CEDAR-RT: complete GPU experiment report", "", completion, "",
        f"Recorded experiment: `{bundle['run_name']}`. All final measurements and earlier recorded attempts are stored together in [data/](data/README.md).", "",
        "## Setup", "",
        'The four GPU backends are **PyTorch (`Torch`)**, **ONNX Runtime CUDA (`CUDA`)**, **ONNX Runtime TensorRT (`ORT TRT`)**, and **direct TensorRT (`TRT`)**. PyTorch and ONNX Runtime CUDA use FP32/FP16; both TensorRT backends use FP16/INT8. These form eight backend/precision settings. One experimental configuration is one model, batch size, and backend/precision setting. **No CPU measurements are included.**', "",
        "**All experiments in this report run on the PC GPU**, in Conda `torch270_py312_cuda`; no Jetson deployment tests are reported. Checkpoint hashes identify the exact model edition used in each run.", "",
        "The [README environment table](../README.md#installation) lists tested/recommended versions. Compatible general runtime libraries can be used. For raw-I/Q conversion, preserve preprocessing library versions and parameters: even small pixel changes can invalidate pretrained predictions. The recorded RT experiments use the supplied PNGs directly.", "",
        f"- Device: **{env.get('gpu','see metadata')}**, {env.get('platform','see metadata')}.",
        f"- Software: Python {env.get('python','see metadata')}; PyTorch {packages.get('torch','see metadata')}; ONNX Runtime {packages.get('onnxruntime-gpu','see metadata')}; TensorRT {packages.get('tensorrt','see metadata')}.",
        f"- Evaluation: **{split} split, {', '.join(map(str,images))} images**. INT8 calibration uses 512 training images.",
        "- Models: YOLOv8, YOLO11 and YOLO26 at nano, small and extra-large sizes; CEDAR-YOLO-n/s/x with a 20 × 20 feature grid. Checkpoint identities are recorded in the experiment metadata.",
        f"- Timing: {first['experiment'].get('protocol',{}).get('rounds',3)} rounds per scope, with separate synchronized latency and throughput loops. Each loop uses the recorded minimum duration and image count; inputs and runtime caches are warmed.",
        f"- Scope: {devices} policies at batches {', '.join(map(str,scope['batches']))}. Original plan entries outside this evaluation remain in the data with `in_scope=False`.", "",
        "| Timing scope | Included work |", "|---|---|",
        "| Model computation | Forward pass, with the input tensor already on the device |",
        "| Detector pipeline | Preprocessing, forward pass and box decoding/NMS from a decoded image |",
        "| Image to detection | PNG reading/decoding and the full detector pipeline; warm file cache |", "",
        "## Metrics", "", "| Metric | Meaning |", "|---|---|",
        "| Inference time per frame (s/frame) ↓ | Inverse of mean throughput across the three rounds. At batch > 1, this is time normalized by throughput, not request latency. |",
        "| Overlap F1 ↑ | Frame-level radar/LTE frequency-overlap F1, using the MDI rule: any positive-width frequency intersection. Detections use confidence 0.25; overlap is the positive class. |",
        "| mAP@0.5 ↑ | Mean LTE/radar box AP at IoU 0.5; confidence-greedy matching and 101 recall points. |",
        "| mAP@0.5:0.95 ↑ | Mean AP over IoU thresholds 0.50 to 0.95; retained in all detailed tables. |",
        "| p50 / p95 / p99 | Per-batch request latency percentiles in milliseconds, averaged across rounds. |", "",
        quality_note + " Repeated quality columns across timing scopes refer to that same evaluation.", "",
        "## Results", "",
        "The three figures below compare inference speed, overlap F1 and mAP@0.5 at batch 1. "
        "The detailed tables cover all evaluated batch sizes and timing scopes.", ""]
    overview_scope = overview_files[0][0].stem.split("_b")[0].removeprefix("overview_") if overview_files else "model_compute"
    overview = overview_sections(rows, overview_files, output, overview_scope)
    parts.extend(overview)
    parts.extend(["## Completion and data audit", "", "| Batch | Completed GPU configurations | Remaining |", "|---:|---:|---:|"])
    for batch in scope["batches"]:
        group = [r for r in selected_routes if int(r["batch"]) == batch]
        n = sum(r["status"] == "passed" for r in group)
        parts.append(f"| {batch} | {n}/{len(group)} | {len(group)-n} |")
    parts.extend(["", f"The final tables contain **{len(rows)} model × batch × policy × timing-scope aggregates**. Each aggregate is checked against its original rounds and protocol signatures.", "",
        f"Earlier attempts are retained in `data/history.jsonl` ({audit.get('history_attempts','see manifest')} distinct archived attempts). They are excluded from the final averages. The original full experiment plan and canceled entries remain traceable in `data/routes.csv` and `data/metadata.json`.", "",
        "## All model results", "", "| Model | Final configurations | Detailed results |", "|---|---:|---|"])
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
            parts.extend(["| Batch | Policy | Time (s/frame) ↓ | Overlap F1 ↑ | mAP@0.5 ↑ | mAP@0.5:0.95 ↑ | p50 (ms/batch) | p95 (ms/batch) | p99 (ms/batch) |",
                          "|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
            for row in sorted((r for r in rows if r["model"]==model and r["scope"]==timing_scope),key=lambda r:(int(r["batch"]),MODES.index(r["mode"]))):
                fields = [("inference_time_per_frame_s",6),("overlap_f1",4),
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
    summary = ["## Results", "", completion, "",
        f"The experiments evaluate {len(scope['models'])} pretrained detectors on an **{env.get('gpu','recorded GPU')}** PC. "
        f"The figures show **batch 1, {SCOPE_LABELS.get(overview_scope,overview_scope).lower()}**; "
        f"quality is evaluated on the same **{', '.join(map(str,images))} {split} images**. "
        "INT8 calibration uses 512 training images. Each policy uses the same source checkpoint for its model.", "", quality_note, "",
        *overview,
        "**Deployment assessment.** TensorRT FP16 provides the most consistent combination of faster model "
        "execution and preserved detection quality in this testbed. INT8 can further reduce computation time "
        "for selected models, with quality checked for the selected configuration. The full report separates "
        "model computation, the detector pipeline, and PNG-to-detection execution so application costs can be assessed alongside kernel speed.", "",
        "[Complete report](REPORT.md) · [Combined measurements](data/summary.csv) · [Data guide](data/README.md)", ""]
    (output / "SUMMARY.md").write_text("\n".join(summary),encoding="utf-8")
    guide = """# Consolidated experiment data

This is the single data location used by `python analyze_results.py`. Files are combined across all models, batches and policies; no per-model result directories are needed.

| File | Contents |
|---|---|
| `summary.csv` | Final full-precision aggregates: model, batch, policy and timing scope; time per frame, throughput, F1, mAP and latency percentiles. |
| `rounds.csv` | Every original row from the latest compact results, including unrun plan placeholders. Filter `status == passed` and a nonempty `scope` for measurements. |
| `routes.csv` | Original full-plan states plus `in_scope`; use `in_scope == True` for the final GPU evaluation. |
| `overlap_evaluations.csv` | Quality-only re-evaluation: overlap confusion counts, weights/data identity, evaluation time and diagnostic mAP. Original timing and mAP are retained. |
| `metadata.json` | All model metadata, experiment protocols, environments, hashes, calibration records and the explicit final report scope. |
| `dataset.json` | Dataset membership, split labels and hashes; no input images. |
| `history.jsonl` | Earlier recorded attempts, including failures and available partial measurements; these are not extra repetitions of the final evaluation. |
| `manifest.json` | SHA-256 checksums, source audit and analysis-code hashes. |

`inference_time_per_frame_s = 1 / mean(images_per_second)` across the three timing rounds. `overlap_f1` is the frame-level radar/LTE frequency-overlap F1, computed from pooled confusion counts using the same definition as MDI. Historical `macro_f1`, `lte_f1` and `radar_f1` fields remain in the data for traceability and are not plotted. Quality is evaluated once per configuration and repeated alongside each timing scope.

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
