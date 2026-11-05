"""PNG demo, quality evaluation and explicitly scoped performance measurements."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import Dataset, load_labels, load_png, to_tensor
from .io import environment, write_json
from .metrics import summarize
from .postprocess import detections
from .runtime import synchronize


@torch.inference_mode()
def infer(backend, images: list[np.ndarray], confidence=.25, iou=.45, max_det=300):
    valid = len(images)
    if not 1 <= valid <= backend.batch:
        raise ValueError("Image batch exceeds backend batch")
    padded = images + [images[-1]] * (backend.batch-valid)
    tensor = to_tensor(padded, backend.device, backend.dtype)
    return [d.cpu().numpy() for d in detections(backend(tensor), backend.meta["layout"],
                                              confidence, iou, max_det)[:valid]]


def predict(backend, paths: list[Path], output: Path, confidence=.25, iou=.45, max_det=300):
    validate_demo_output(paths, output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for start in range(0, len(paths), backend.batch):
        chunk = paths[start:start+backend.batch]
        images = [load_png(path) for path in chunk]
        for path, rgb, det in zip(chunk, images, infer(backend, images, confidence, iou, max_det)):
            annotated = Image.fromarray(rgb)
            draw = ImageDraw.Draw(annotated)
            for x1, y1, x2, y2, score, cls in det:
                label = backend.meta["classes"][int(cls)]
                color = "lime" if cls == 0 else "orange"
                draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
                draw.text((x1, max(0, y1-12)), f"{label} {score:.3f}", fill=color)
            annotated.save(output / path.name)
            records.append({"image": path.name, "detections_xyxy_score_class": det.tolist()})
    result = {"model": backend.meta, "confidence": confidence, "nms_iou": iou,
              "records": records, "environment": environment()}
    write_json(output / "predictions.json", result)
    return result


def validate_demo_output(paths: list[Path], output: Path) -> None:
    if len({p.name.casefold() for p in paths}) != len(paths):
        raise ValueError("Duplicate filenames would overwrite demo output")
    sources = {p.resolve() for p in paths}
    for path in [output / p.name for p in paths] + [output / "predictions.json"]:
        if path.resolve() in sources or path.exists():
            raise FileExistsError(f"Refusing to overwrite an input or existing demo artifact: {path}")


def validate_dataset_backend(backend, dataset: Dataset) -> None:
    calibration = backend.meta.get("calibration")
    if calibration:
        if calibration["profile_sha256"] != dataset.manifest["profile_sha256"]:
            raise ValueError("Dataset rendering profile differs from INT8 calibration")
        if calibration["dataset_sha256"] != dataset.manifest["records_sha256"]:
            raise ValueError("Dataset differs from the frozen INT8 calibration experiment")


def evaluate(backend, dataset: Dataset, confidence=.25, iou=.45, max_det=300, progress=None) -> dict:
    if not .001 <= confidence <= 1:
        raise ValueError("Operating confidence must be in [.001, 1]")
    validate_dataset_backend(backend, dataset)
    pairs = []
    for start in range(0, len(dataset.records), backend.batch):
        records = dataset.records[start:start+backend.batch]
        images = [load_png(dataset.path(r["image"])) for r in records]
        predictions = infer(backend, images, confidence, iou, max_det)
        pairs.extend((pred, load_labels(dataset.path(r["label"])))
                     for pred, r in zip(predictions, records))
        if progress:
            progress({"completed": min(start+backend.batch, len(dataset.records)), "total": len(dataset.records)})
    return {"kind": "evaluation", "model": backend.meta,
            "dataset_sha256": dataset.manifest["records_sha256"],
            "profile_sha256": dataset.manifest["profile_sha256"],
            "split": dataset.records[0]["split"], "collection_confidence": confidence,
            "nms_iou": iou, "max_det": max_det, "metrics": summarize(pairs, confidence),
            "environment": environment()}


@torch.inference_mode()
def benchmark(backend, dataset: Dataset, scope: str, warmup: int = 50,
              min_images: int = 1000, seconds: float = 30, rounds: int = 3,
              confidence=.25, iou=.45, max_det=300, pool_size=64, progress=None) -> dict:
    """Latency synchronizes each request. Throughput uses a separate measured loop.

    model_compute uses an input ring resident on the selected CPU/GPU device.
    All routes use synchronized wall time, including ORT scheduling and fallback.
    Disk reads use the OS page cache; no claim of cold-storage performance is made.
    """
    if scope not in ("model_compute", "detector_pipeline", "image_to_detection"):
        raise ValueError(f"Unknown timing scope: {scope}")
    validate_dataset_backend(backend, dataset)
    if min(warmup, min_images, rounds, pool_size) < 1 or seconds <= 0:
        raise ValueError("Benchmark counts and duration must be positive")
    records = dataset.records[:min(pool_size, len(dataset.records))]
    paths = [dataset.path(r["image"]) for r in records]
    # CPU/GPU pools are bounded. Every request is a full batch, including repeats.
    groups = [[paths[(i+j) % len(paths)] for j in range(backend.batch)]
              for i in range(0, len(paths), backend.batch)]
    decoded = [[load_png(p) for p in group] for group in groups] if scope != "image_to_detection" else None
    tensors = [to_tensor(group, backend.device, backend.dtype) for group in decoded] if scope == "model_compute" else None

    def request(index):
        slot = index % len(groups)
        if scope == "model_compute":
            return backend(tensors[slot])
        images = decoded[slot] if decoded is not None else [load_png(p) for p in groups[slot]]
        return infer(backend, images, confidence, iou, max_det)

    if progress:
        progress({"measurement": "warmup"})
    for i in range(warmup):
        request(i)
    synchronize(backend)
    gpu = backend.device.type == "cuda"
    if gpu:
        torch.cuda.reset_peak_memory_stats(backend.device)
    results = []
    for round_index in range(rounds):
        latencies, completed = [], 0
        if progress:
            progress({"round": round_index+1, "rounds": rounds, "measurement": "latency"})
        start_wall = time.perf_counter()
        while completed*backend.batch < min_images or time.perf_counter()-start_wall < seconds:
            begin_wall = time.perf_counter()
            request(completed)
            synchronize(backend)
            elapsed_ms = (time.perf_counter()-begin_wall)*1000
            latencies.append(elapsed_ms)
            completed += 1
        synchronize(backend)
        if progress:
            progress({"round": round_index+1, "rounds": rounds, "measurement": "throughput"})
        started, requests = time.perf_counter(), 0
        # Bound the async queue to avoid measuring an unbounded enqueue backlog.
        while requests*backend.batch < min_images or time.perf_counter()-started < seconds:
            request(requests)
            requests += 1
            if scope != "model_compute" or requests % 16 == 0:
                synchronize(backend)
        synchronize(backend)
        wall = time.perf_counter()-started
        results.append({"latency_requests": completed, "latency_ms_mean": float(np.mean(latencies)),
                        "latency_ms_p50": float(np.percentile(latencies, 50)),
                        "latency_ms_p95": float(np.percentile(latencies, 95)),
                        "latency_ms_p99": float(np.percentile(latencies, 99)),
                        "throughput_images": requests*backend.batch, "throughput_seconds": wall,
                        "images_per_second": requests*backend.batch/wall})
    return {"kind": "benchmark", "protocol_version": 2, "latency_clock": "synchronized_wall",
            "model": backend.meta, "scope": scope,
            "warmup_requests": warmup, "minimum_images": min_images, "minimum_seconds": seconds,
            "pool_images": len(paths), "pool_repeats": True, "rounds": results,
            "confidence": confidence, "nms_iou": iou, "max_det": max_det,
            "dataset_sha256": dataset.manifest["records_sha256"],
            "profile_sha256": dataset.manifest["profile_sha256"],
            "split": dataset.records[0]["split"], "environment": environment(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated() if gpu else None,
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved() if gpu else None,
            "memory_note": "PyTorch CUDA allocator only; excludes ORT/TRT and CPU allocations. Jetson memory is shared.",
            "timing_note": "separate latency/throughput loops; no drawing, metrics or output I/O; warm file cache"}
