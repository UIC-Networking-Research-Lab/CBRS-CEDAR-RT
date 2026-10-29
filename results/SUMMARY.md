## Results

**288/288 GPU routes completed: 12 models, batches 1, 4, 8, 8 deployment policies.**

**NVIDIA GeForce RTX 2070 · batch 1 · model computation.** Quality uses the **validation split (846 images)**.

**Inference time per frame (s/frame) ↓ · Detection F1 ↑ · mAP@0.5 ↑**. Time is the inverse of mean throughput; F1 averages LTE and radar box-detection F1.

**Example: YOLOv8x, same checkpoint and validation images.**

| Policy | Time (s/frame) ↓ | Detection F1 ↑ | mAP@0.5 ↑ |
|---|---:|---:|---:|
| PyTorch FP32 | 0.0412 | 0.9976 | 0.9950 |
| ORT TRT FP16 | 0.0109 | 0.9976 | 0.9950 |
| ORT TRT INT8 | 0.0087 | 0.9979 | 0.9950 |

<details>
<summary>All 12 models: inference time, F1 and mAP comparisons</summary>

![Policy comparison](figures/overview_model_compute_b001_inference_time_per_frame_s.png)

![Policy comparison](figures/overview_model_compute_b001_macro_f1.png)

![Policy comparison](figures/overview_model_compute_b001_map50.png)

</details>

FP16 and INT8 can reduce model computation time; compare detection quality alongside speed. The full report also includes preprocessing and image decoding.

[Complete report](REPORT.md) · [Combined measurements](data/summary.csv) · [Data guide](data/README.md)
