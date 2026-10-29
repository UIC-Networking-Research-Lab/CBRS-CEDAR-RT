"""Export a CBRS checkpoint to ONNX; edit defaults below or use CLI options."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL = "cbrs_yolo8n"
WEIGHTS = None                 # None: resolve MODEL from the local weights manifest.
OUTPUT = None                  # None: exports/<model>/<shape>_<precision>.onnx.
BATCH_SIZE = None              # None: dynamic batch; an integer fixes the batch.
PRECISION = "fp32"             # fp32, fp16 or calibrated int8.
DEVICE = "cpu"                 # fp16 export requires cuda.
DATA_DIR = "data/yolo_split"   # INT8 calibration PNGs must match the training profile.
DATASET_MANIFEST = "artifacts/dataset.json"
WEIGHTS_MANIFEST = "artifacts/weights/manifest.json"
CALIBRATION_SAMPLES = 512
CALIBRATION_SEED = 42
CPU_THREADS = 1

sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def absolute(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def parser():
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    cli.add_argument("--model", default=MODEL)
    cli.add_argument("--weights", default=WEIGHTS)
    cli.add_argument("--output", default=OUTPUT)
    shape = cli.add_mutually_exclusive_group()
    shape.add_argument("--batch", type=int, default=BATCH_SIZE, help="Fixed batch size")
    shape.add_argument("--dynamic", dest="batch", action="store_const", const=None, help="Dynamic batch")
    cli.add_argument("--precision", choices=("fp32", "fp16", "int8"), default=PRECISION)
    cli.add_argument("--device", choices=("cpu", "cuda"), default=DEVICE)
    cli.add_argument("--data", default=DATA_DIR)
    cli.add_argument("--manifest", default=DATASET_MANIFEST)
    cli.add_argument("--weights-manifest", default=WEIGHTS_MANIFEST)
    cli.add_argument("--samples", type=int, default=CALIBRATION_SAMPLES)
    cli.add_argument("--seed", type=int, default=CALIBRATION_SEED)
    cli.add_argument("--threads", type=int, default=CPU_THREADS)
    cli.add_argument("--dry-run", action="store_true", help="Print the conversion plan without writing files")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if not re.fullmatch(r"cbrs_yolo(?:8|11|26)[nsx]|sig_yolo_v8[nsx]_f\d+t\d+_c\d+", args.model):
            raise ValueError(f"Unsupported model identifier: {args.model}")
        if args.precision not in ("fp32", "fp16", "int8") or args.device not in ("cpu", "cuda"):
            raise ValueError("Unsupported precision or device")
        if (args.batch is not None and args.batch < 1) or args.samples < 1 or args.threads < 1:
            raise ValueError("Batch, calibration samples and threads must be positive")
        if args.precision == "fp16" and args.device != "cuda":
            raise ValueError("FP16 export requires --device cuda")
        expected_sha = None
        if args.weights:
            weights = absolute(args.weights)
        else:
            import json
            entries = json.loads(absolute(args.weights_manifest).read_text(encoding="utf-8"))
            entry = next((item for item in entries if item["model_id"] == args.model), None)
            if entry is None:
                raise ValueError(f"Model not in weights manifest: {args.model}")
            weights = absolute(entry["path"])
            expected_sha = entry["sha256"]
            if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                raise ValueError("Invalid checkpoint SHA-256 in weights manifest")
        shape = "dynamic" if args.batch is None else f"batch_{args.batch:03d}"
        output = absolute(args.output or f"exports/{args.model}/{shape}_{args.precision}.onnx")
        if output.suffix.lower() != ".onnx":
            raise ValueError("Output must have a .onnx extension")
        print(f"ONNX export | {args.model} | {shape} | {args.precision} | {args.device}", flush=True)
        print(f"  Weights: {weights}\n  Output:  {output}", flush=True)
        if args.precision == "int8":
            print(f"  INT8 calibration: {args.samples} training PNGs; seed {args.seed}", flush=True)
        if args.dry_run:
            print("Plan only; files and runtime were not checked.", flush=True)
            return 0
        if expected_sha:
            from cbrs_quant.io import digest
            if digest(weights) != expected_sha:
                raise ValueError("Checkpoint does not match the weights manifest")
        import torch
        torch.set_num_threads(args.threads)
        dataset = None
        if args.precision == "int8":
            from cbrs_quant.data import Dataset
            dataset = Dataset(absolute(args.data), absolute(args.manifest), "train")

        def progress(done, total):
            if done == 1 or done % 50 == 0 or done == total:
                print(f"  Calibration: {done}/{total}", flush=True)

        if args.batch is None:
            from cbrs_quant.dynamic_export import export_dynamic
            meta = export_dynamic(args.model, weights, output, args.precision, args.device,
                                  dataset, args.samples, args.seed, progress)
        else:
            from cbrs_quant.export import export_onnx
            callback = (lambda event: progress(event["completed"], event["total"]) if "total" in event else None)
            meta = export_onnx(args.model, weights, output, args.batch, dataset, args.samples, args.seed,
                               "fp16" if args.precision == "fp16" else "fp32", args.device, callback)
        print(f"Done: {output}\nMetadata: {output}.json\nSHA-256: {meta['sha256']}", flush=True)
        if args.batch is None:
            print("Dynamic batch requires a compatible runtime; this project's built-in backends currently require --batch.",
                  flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"Export failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
