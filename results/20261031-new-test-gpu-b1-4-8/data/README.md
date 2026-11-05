# Current test experiment data

All records in this directory belong to the current test sweep: 12 models, batches 1/4/8 and eight GPU policies (288 configurations). The fixed test split contains 848 PNGs. No CPU or historical validation results are included in the reported aggregates.

| File | Contents |
|---|---|
| `summary.csv` | 864 final model/batch/policy/scope aggregates, including time per frame, overlap F1, mAP and per-batch latency percentiles. |
| `rounds.csv` | Original timing rows and the quality measurements attached to them. |
| `routes.csv` | Configuration completion states and report membership. |
| `metadata.json` | Checkpoint hashes, runtime versions, evaluation protocol, export and calibration metadata. |
| `dataset.json` | Fixed split membership and SHA-256 hashes of PNGs and labels. |
| `history.jsonl` | Earlier attempts, excluded from final averages. |
| `manifest.json` | File checksums and analysis provenance. |

`inference_time_per_frame_s = 1 / mean(images_per_second)` across three rounds; this is reciprocal throughput, not mean request latency. `overlap_f1` uses pooled frame-level TP/FP/FN and the CEDAR-MDI frequency-overlap rule. `macro_f1`, `lte_f1` and `radar_f1` are diagnostic box metrics and are not the F1 shown in figures. mAP uses same-class IoU matching and interpolated trapezoidal AP after confidence filtering at 0.25.

Quality is evaluated once per configuration. Its repetition across three timing scopes and rounds does not represent independent quality trials. Filter one scope when comparing quality across configurations.

From the repository root:

```python
import csv
from pathlib import Path
with Path("results/20261031-new-test-gpu-b1-4-8/data/summary.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
```

Historical absolute paths in metadata identify the recorded run; report generation uses the files in this directory. Download images and checkpoints from the separate assets package to rerun inference.
