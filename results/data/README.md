# Consolidated experiment data

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
