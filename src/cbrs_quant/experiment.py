"""One reproducible local experiment; retain model files even when a route fails."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import shutil
import time

import torch

from .compare import compare
from .data import Dataset
from .engine import build_engine
from .export import export_onnx
from .io import digest, environment, write_json
from .runner import benchmark, evaluate, predict
from .runtime import create_backend, is_cpu_mode, synchronize
from .smoke import smoke


def experiment(model_id: str, weights: Path, data: Path, manifest: Path, output: Path,
               modes: list[str], quick=False, batch=1, samples=512, threads=1,
               allow_cpu_fallback=False, split="test") -> dict:
    if len(set(modes)) != len(modes) or not modes or batch < 1 or threads < 1:
        raise ValueError("Modes must be unique; batch and threads must be positive")
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "models"
    model_dir.mkdir()
    retained_pt = model_dir / "checkpoint.pt"
    shutil.copyfile(weights, retained_pt)
    if digest(weights) != digest(retained_pt):
        raise RuntimeError("Checkpoint copy verification failed")
    shutil.copyfile(manifest, output / "dataset.json")
    torch.set_num_threads(threads)
    report = {"schema": 2, "model_id": model_id, "smoke_only": quick, "requested_modes": modes,
              "environment": environment(), "results": {}, "batch": batch, "split": split,
              "threads": threads, "allow_cpu_fallback": allow_cpu_fallback, "status": "running"}
    optimization_level = 0 if quick else 3
    report["builder_optimization_level"] = optimization_level
    write_json(output / "experiment.json", report)
    try:
        dataset = Dataset(data, manifest, split)
        if quick:
            dataset.records = dataset.records[:8]
        paths = [dataset.path(r["image"]) for r in dataset.records[:3]]
        device = "cpu" if all(is_cpu_mode(m) for m in modes) else "cuda"
        graphs = {"fp32": model_dir / "model.fp32.onnx"}
        export_onnx(model_id, retained_pt, graphs["fp32"], batch=batch, device_name=device)
        if "ort_cuda_fp16" in modes:
            graphs["fp16"] = model_dir / "model.fp16.onnx"
            export_onnx(model_id, retained_pt, graphs["fp16"], batch=batch, precision="fp16")
        if any(m.endswith("int8") for m in modes):
            train = Dataset(data, manifest, "train")
            graphs["int8"] = model_dir / "model.int8.onnx"
            export_onnx(model_id, retained_pt, graphs["int8"], batch=batch, dataset=train,
                        samples=min(4, len(train.records)) if quick else samples)
        scopes = ("model_compute", "detector_pipeline", "image_to_detection")
        comparisons = {scope: [] for scope in scopes}
        for mode in modes:
            backend = None
            try:
                engine = None
                graph = graphs["fp16" if mode == "ort_cuda_fp16" else "int8" if mode.endswith("int8") else "fp32"]
                if mode.startswith("trt_"):
                    engine = model_dir / f"{mode}.engine"
                    build_engine(graph, engine, precision=mode.removeprefix("trt_"),
                                 optimization_level=optimization_level)
                context = nullcontext() if is_cpu_mode(mode) else torch.cuda.stream(torch.cuda.Stream())
                with context:
                    started = time.perf_counter()
                    backend = create_backend(mode, model_id, retained_pt, engine, batch, graph,
                                             output / "ort_cache", allow_cpu_fallback, threads,
                                             optimization_level=optimization_level)
                    synchronize(backend)
                    load_seconds = time.perf_counter() - started
                    result_dir = output / "results" / mode
                    parity = None if mode.endswith("int8") else retained_pt
                    write_json(result_dir / "smoke.json", smoke(backend, paths, parity))
                    predict(backend, paths, result_dir / "demo")
                    quality = evaluate(backend, dataset)
                    quality.update(smoke_only_subset=quick, load_seconds=load_seconds)
                    write_json(result_dir / "evaluation.json", quality)
                    for scope in scopes:
                        timing = {"warmup": 2, "min_images": 4, "seconds": .1,
                                  "rounds": 1, "pool_size": 4} if quick else {}
                        measured = benchmark(backend, dataset, scope, **timing)
                        measured.update(smoke_only_subset=quick, load_seconds=load_seconds)
                        path = result_dir / f"{scope}.json"
                        write_json(path, measured)
                        comparisons[scope].append(path)
                    report["results"][mode] = {"status": "passed", "metrics": quality["metrics"]}
            except Exception as exc:
                # Keep testing independent requested routes; never count failure as success.
                report["results"][mode] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            finally:
                try:
                    if backend is not None:
                        synchronize(backend)
                    backend = None
                    if not is_cpu_mode(mode) and torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    backend = None
                    report["results"].setdefault(mode, {"status": "failed"})["cleanup_error"] = str(exc)
                    report["results"][mode]["status"] = "failed"
            write_json(output / "experiment.json", report)
        for scope, files in comparisons.items():
            if files:
                compare(files, output / "comparison" / scope)
        report["status"] = "passed" if all(r["status"] == "passed" for r in report["results"].values()) else "failed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # No temporary model directory is deleted. PT and ONNX survive partial failures.
        report["retained_artifacts"] = [{"path": p.relative_to(output).as_posix(), "sha256": digest(p),
                                          "bytes": p.stat().st_size}
                                         for p in sorted(model_dir.iterdir()) if p.is_file()]
        write_json(output / "experiment.json", report)
    return report
