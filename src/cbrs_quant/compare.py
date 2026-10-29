"""Compare only reports with matching models, input contracts and timing scope."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .io import read_json, write_json


def compare(paths: list[Path], output: Path) -> list[dict]:
    reports = [read_json(p) for p in paths]
    if not reports or any(r.get("kind") != "benchmark" for r in reports):
        raise ValueError("compare requires benchmark JSON reports")
    fields = ("protocol_version", "latency_clock", "dataset_sha256", "profile_sha256", "scope", "split", "confidence", "nms_iou", "max_det",
              "pool_images", "warmup_requests", "minimum_images", "minimum_seconds")
    reference = reports[0]
    if any(r.get("protocol_version") != 2 for r in reports):
        raise ValueError("Use current protocol-v2 reports; old CUDA-event latency is not comparable")
    trt_reports = [r for r in reports if r["model"].get("route") in ("B", "C")]
    levels = {r["model"].get("builder_optimization_level") for r in trt_reports}
    if len(levels) > 1 or (trt_reports and None in levels):
        raise ValueError("TensorRT builder optimization levels differ or are unknown")
    for report in reports[1:]:
        if any(report[k] != reference[k] for k in fields):
            raise ValueError("Benchmark protocols differ; compare one scope/protocol at a time")
        if any(report["model"][k] != reference["model"][k]
               for k in ("model_id", "weights_sha256", "batch")):
            raise ValueError("Benchmark checkpoints or batches differ")
        if len(report["rounds"]) != len(reference["rounds"]):
            raise ValueError("Benchmark round counts differ")
        for key in ("gpu", "compute_capability", "l4t", "cuda_build", "python", "platform", "machine",
                    "torch_threads", "torch_interop_threads", "packages", "nvidia_tf32_override"):
            if report["environment"].get(key) != reference["environment"].get(key):
                raise ValueError("Benchmark hardware/environment differs")
    modes = [r["model"]["mode"] for r in reports]
    if len(set(modes)) != len(modes):
        raise ValueError("Duplicate modes; merge repeated trials explicitly")
    baseline = next((r for r in reports if r["model"]["mode"] == "torch_cuda_fp32"), None)
    baseline_rate = np.mean([x["images_per_second"] for x in baseline["rounds"]]) if baseline else None
    rows = []
    for report in reports:
        trials = report["rounds"]
        rate = float(np.mean([x["images_per_second"] for x in trials]))
        rows.append({"model": report["model"]["model_id"], "mode": report["model"]["mode"],
                     "route": report["model"].get("route"), "device": report["model"].get("device"),
                     "precision": report["model"].get("precision"),
                     "ort_threads": report["model"].get("ort_threads"),
                     "builder_optimization_level": report["model"].get("builder_optimization_level"),
                     "provider_nodes": str(report["model"].get("provider_audit", {}).get("node_events", {})),
                     "batch": report["model"]["batch"], "scope": report["scope"],
                     "images_per_second_mean": rate,
                     "images_per_second_std": float(np.std([x["images_per_second"] for x in trials])),
                     "p50_ms_mean": float(np.mean([x["latency_ms_p50"] for x in trials])),
                     "p95_ms_mean": float(np.mean([x["latency_ms_p95"] for x in trials])),
                     "speedup_vs_fp32": rate/baseline_rate if baseline_rate else None})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", rows)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
