"""Explicit commands; optional dependencies are loaded only by the selected action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from .config import MODES, DEFAULT_MODES


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="python cli.py", description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Record actual container and GPU versions")
    doctor.add_argument("--output", type=Path)
    doctor.add_argument("--require", nargs="*", choices=("cuda", "tensorrt", "cpu"))
    validate = commands.add_parser("validate-inputs", help="Validate and hash a YOLO PNG dataset")
    validate.add_argument("--data", type=Path, required=True)
    validate.add_argument("--profile", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)

    def model_args(p):
        p.add_argument("--model", required=True)
        p.add_argument("--weights", type=Path)
        p.add_argument("--batch", type=int, default=1)

    def data_args(p, split=True):
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--manifest", type=Path, required=True)
        if split:
            p.add_argument("--split", choices=("val", "test"), default="test")

    for command in ("export", "quantize"):
        p = commands.add_parser(command, help="Export static FP32 or INT8 Q/DQ ONNX")
        model_args(p)
        p.add_argument("--output", type=Path, required=True)
        if command == "quantize":
            data_args(p, split=False)
            p.add_argument("--samples", type=int, default=512)
            p.add_argument("--seed", type=int, default=42)
        else:
            p.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
            p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    build = commands.add_parser("build", help="Build a TensorRT engine on the target Jetson")
    build.add_argument("--onnx", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--workspace-gib", type=float, default=2)
    build.add_argument("--precision", choices=("fp32", "fp16", "int8"))
    build.add_argument("--optimization-level", type=int, choices=range(6), default=3)

    for command in ("predict", "evaluate", "benchmark", "smoke"):
        p = commands.add_parser(command)
        model_args(p)
        p.add_argument("--mode", choices=MODES, required=True)
        p.add_argument("--engine", type=Path)
        p.add_argument("--onnx", type=Path)
        p.add_argument("--cache-dir", type=Path, default=Path("artifacts/ort_cache"))
        p.add_argument("--allow-cpu-fallback", action="store_true")
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--workspace-gib", type=float, default=2)
        p.add_argument("--optimization-level", type=int, choices=range(6), default=3)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--confidence", type=float, default=.25)
        p.add_argument("--iou", type=float, default=.45)
        p.add_argument("--max-det", type=int, default=300)
        if command in ("predict", "smoke"):
            p.add_argument("--input", type=Path, required=True, help="PNG or directory of PNGs")
        else:
            data_args(p)
        if command == "smoke":
            p.add_argument("--reference-weights", type=Path, help="Compare with PyTorch FP32 on the same device")
            p.add_argument("--max-relative-l2", type=float, default=.05)
        if command == "benchmark":
            p.add_argument("--scope", choices=("model_compute", "detector_pipeline", "image_to_detection"), required=True)
            p.add_argument("--warmup", type=int, default=50)
            p.add_argument("--min-images", type=int, default=1000)
            p.add_argument("--seconds", type=float, default=30)
            p.add_argument("--rounds", type=int, default=3)
            p.add_argument("--pool-size", type=int, default=64)
    comparison = commands.add_parser("compare")
    comparison.add_argument("reports", nargs="+", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    archive = commands.add_parser("archive", help="Save PT/ONNX/engine files with verified provenance")
    archive.add_argument("--weights", type=Path, required=True)
    archive.add_argument("--onnx", nargs="+", type=Path, required=True)
    archive.add_argument("--engine", nargs="*", type=Path, default=[])
    archive.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("experiment", help="Run selected routes and retain PT/ONNX/engines/results")
    model_args(p)
    data_args(p)
    p.set_defaults(split="val")
    p.add_argument("--modes", nargs="+", choices=MODES, default=DEFAULT_MODES)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true", help="Short timings, 8 eval images and 4 calibration images")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--allow-cpu-fallback", action="store_true")
    for command in ("from-iq", "from-sigmf"):
        p = commands.add_parser(command, help="Offline reproduction only")
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--profile", type=Path, required=True)
        p.add_argument("--allow-version-mismatch", action="store_true")
        if command == "from-sigmf":
            p.add_argument("--meta", type=Path, required=True)
            p.add_argument("--cbrs-rate-override", action="store_true")
    pixels = commands.add_parser("verify-png")
    pixels.add_argument("--actual", type=Path, required=True)
    pixels.add_argument("--expected", type=Path, required=True)
    pixels.add_argument("--tolerance", type=int, default=0)
    return cli


def run(args):
    from .io import environment, write_json
    command = args.command
    if command == "doctor":
        result = environment()
        if args.output:
            write_json(args.output, result)
        names = {"cuda": "CUDAExecutionProvider", "tensorrt": "TensorrtExecutionProvider", "cpu": "CPUExecutionProvider"}
        present = (result["ort"] or {}).get("available_providers", [])
        if any(names[p] not in present for p in (args.require or [])):
            raise RuntimeError(f"ORT provider preflight failed. Available: {present}. See the Installation section of README.md")
        return result
    if command == "validate-inputs":
        from .data import make_manifest
        result = make_manifest(args.data, args.profile, args.output)
        return {"records": len(result["records"]), "manifest": str(args.output)}
    if command in ("export", "quantize"):
        from .data import Dataset
        from .export import export_onnx
        if args.weights is None:
            raise ValueError("Export requires --weights")
        dataset = Dataset(args.data, args.manifest, "train") if command == "quantize" else None
        result = export_onnx(args.model, args.weights, args.output, args.batch, dataset,
                             getattr(args, "samples", 512), getattr(args, "seed", 42),
                             getattr(args, "precision", "fp32"), getattr(args, "device", "cuda"))
        return {"artifact": str(args.output), "sha256": result["sha256"], "qdq_nodes": result["qdq_nodes"]}
    if command == "build":
        from .engine import build_engine
        result = build_engine(args.onnx, args.output, args.workspace_gib, args.precision, args.optimization_level)
        return {"artifact": str(args.output), "sha256": result["sha256"], "seconds": result["build_seconds"]}
    if command == "compare":
        from .compare import compare
        return compare(args.reports, args.output)
    if command == "archive":
        from .archive import archive_models
        return archive_models(args.weights, args.onnx, args.engine, args.output)
    if command == "experiment":
        from .experiment import experiment
        if args.weights is None:
            raise ValueError("Experiment requires --weights")
        result = experiment(args.model, args.weights, args.data, args.manifest, args.output,
                            args.modes, args.smoke, args.batch, args.samples, args.threads,
                            args.allow_cpu_fallback, args.split)
        if result["status"] != "passed":
            raise RuntimeError(f"Some routes failed; PT/ONNX and details retained in {args.output}")
        return {"output": str(args.output), "status": result["status"], "modes": list(result["results"])}
    if command in ("from-iq", "from-sigmf"):
        from .conversion import convert
        return convert(args.input, args.output, args.profile, getattr(args, "meta", None),
                       args.allow_version_mismatch, getattr(args, "cbrs_rate_override", False))
    if command == "verify-png":
        from .conversion import verify_png
        result = verify_png(args.actual, args.expected, args.tolerance)
        if not result["passed"]:
            raise ValueError(f"Pixel verification failed: {result}")
        return result
    from .runtime import create_backend, synchronize
    import torch
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    import time
    started = time.perf_counter()
    backend = create_backend(args.mode, args.model, args.weights, args.engine, args.batch,
                             args.onnx, args.cache_dir, args.allow_cpu_fallback, args.threads,
                             args.workspace_gib, args.optimization_level)
    synchronize(backend)
    load_seconds = time.perf_counter() - started
    if command in ("predict", "smoke"):
        paths = sorted(args.input.glob("*.png")) if args.input.is_dir() else [args.input]
        if not paths:
            raise ValueError("No PNG inputs")
        if command == "predict":
            from .runner import predict
            result = predict(backend, paths, args.output, args.confidence, args.iou, args.max_det)
            return {"images": len(result["records"]), "output": str(args.output)}
        from .smoke import smoke
        result = smoke(backend, paths, args.reference_weights, args.max_relative_l2,
                       args.confidence, args.iou, args.max_det)
    else:
        from .data import Dataset
        from .runner import benchmark, evaluate
        dataset = Dataset(args.data, args.manifest, args.split)
        if command == "evaluate":
            result = evaluate(backend, dataset, args.confidence, args.iou, args.max_det)
        else:
            result = benchmark(backend, dataset, args.scope, args.warmup, args.min_images,
                               args.seconds, args.rounds, args.confidence, args.iou,
                               args.max_det, args.pool_size)
    result["load_seconds"] = load_seconds
    write_json(args.output, result)
    return {"output": str(args.output), "passed": result.get("passed"), "kind": result.get("kind")}


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        if args.command in ("predict", "evaluate", "benchmark", "smoke") and "cpu" not in args.mode:
            import torch
            from .models import require_cuda
            device = require_cuda()
            # All prep, execution, events and postprocessing share a non-default stream.
            with torch.cuda.stream(torch.cuda.Stream(device=device)):
                result = run(args)
        else:
            result = run(args)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as exc:
        cli.exit(1, f"error: {exc}\n")
