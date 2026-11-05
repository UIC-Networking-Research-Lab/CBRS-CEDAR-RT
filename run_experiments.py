"""Edit the defaults below and Run, or override them with command-line arguments.

Windows, Linux and Jetson use the same entry point. --dry-run never starts a
worker, imports GPU libraries, installs dependencies or writes experiment files.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

# Editable defaults. Relative paths are resolved against this script, not the IDE cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS = ["all"]                       # Or ["cbrs_yolo8n", "sig_yolo_v8n_f20t20_c100"].
BATCH_SIZES = [1, 4, 8]
ROUTES = ["A", "B", "C"]               # D=ORT CPU is available only when explicitly selected.
INCLUDE_PYTORCH = True                 # Include GPU FP32/FP16 references with GPU routes.
MODES = None                          # Optional exact mode list overrides ROUTES/PyTorch.
SMOKE = False                         # True: four calibration PNGs, eight eval PNGs, short timings.
DRY_RUN = False                       # True: show the plan without running anything.
DATA_DIR = "data/yolo_split"           # Prepared PNGs and labels inside this project.
DATASET_MANIFEST = "artifacts/dataset.json"
WEIGHTS_MANIFEST = "artifacts/weights/manifest.json"
OUTPUT_DIR = None                      # None: new timestamped directory under results/.
RESUME = False                        # Requires OUTPUT_DIR pointing to an existing run.
RETRY_FAILED = False
MAKEUP_MODE = False                    # True: run only entries from MAKEUP_LIST in the original run.
MAKEUP_LIST = "makeup/experiments.json"
FORCE_MAKEUP = False                   # True: repeat routes already completed for the same list.
CPU_THREADS = 1
PROGRESS_INTERVAL_SECONDS = 10
MINIMUM_FREE_DISK_GIB = 6
MAXIMUM_WORKER_RAM_GIB = 8

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.dont_write_bytecode = True
from cbrs_quant.config import MODES as AVAILABLE_MODES, ROUTE_MODES  # noqa: E402
from cbrs_quant.compact_sweep import main as run_sweep  # noqa: E402


def absolute(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  allow_abbrev=False)
    cli.add_argument("--models", nargs="+", default=MODELS, help="Model IDs, or all")
    cli.add_argument("--batches", nargs="+", type=int, default=BATCH_SIZES)
    cli.add_argument("--routes", nargs="+", choices=ROUTE_MODES, default=ROUTES)
    cli.add_argument("--pytorch", action=argparse.BooleanOptionalAction, default=INCLUDE_PYTORCH)
    cli.add_argument("--modes", nargs="+", choices=AVAILABLE_MODES, default=MODES,
                     help="Exact backend modes; overrides --routes and --pytorch")
    cli.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=SMOKE)
    cli.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=DRY_RUN)
    cli.add_argument("--data", default=DATA_DIR)
    cli.add_argument("--manifest", default=DATASET_MANIFEST)
    cli.add_argument("--weights-manifest", default=WEIGHTS_MANIFEST)
    cli.add_argument("--output", default=OUTPUT_DIR)
    cli.add_argument("--resume", action=argparse.BooleanOptionalAction, default=RESUME)
    cli.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=RETRY_FAILED)
    cli.add_argument("--makeup", action=argparse.BooleanOptionalAction, default=MAKEUP_MODE)
    cli.add_argument("--makeup-list", default=MAKEUP_LIST)
    cli.add_argument("--force-makeup", action=argparse.BooleanOptionalAction, default=FORCE_MAKEUP)
    cli.add_argument("--threads", type=int, default=CPU_THREADS)
    cli.add_argument("--progress-interval", type=float, default=PROGRESS_INTERVAL_SECONDS)
    cli.add_argument("--minimum-free-gib", type=float, default=MINIMUM_FREE_DISK_GIB)
    cli.add_argument("--maximum-worker-gib", type=float, default=MAXIMUM_WORKER_RAM_GIB)
    return cli


def sweep_arguments(args):
    if args.resume and args.output is None:
        raise ValueError("Resume requires --output (or OUTPUT_DIR) for an existing run")
    routes = args.routes
    modes = args.modes
    if modes is None:
        if not routes or len(set(routes)) != len(routes) or any(r not in ROUTE_MODES for r in routes):
            raise ValueError("ROUTES must contain unique values from A, B, C, D")
        modes = [m for route in routes for m in ROUTE_MODES[route]]
        if args.pytorch and any(route != "D" for route in routes):
            modes = ["torch_cuda_fp32", "torch_cuda_fp16", *modes]
    device = "gpu" if all("cpu" not in m for m in modes) else "cpu" if all("cpu" in m for m in modes) else "mixed"
    batches = "-".join(map(str, args.batches))
    output = args.output or f"results/{datetime.now():%Y%m%d-%H%M%S}-test-{device}-b{batches}"
    argv = ["--project-root", str(PROJECT_ROOT), "--output", str(absolute(output)),
            "--data", str(absolute(args.data)), "--manifest", str(absolute(args.manifest)),
            "--weights-manifest", str(absolute(args.weights_manifest)),
            "--models", *args.models, "--batches", *map(str, args.batches), "--modes", *modes,
            "--threads", str(args.threads), "--progress-interval", str(args.progress_interval),
            "--minimum-free-gib", str(args.minimum_free_gib),
            "--maximum-worker-gib", str(args.maximum_worker_gib)]
    for name in ("smoke", "dry_run", "resume", "retry_failed"):
        if getattr(args, name):
            argv.append("--" + name.replace("_", "-"))
    return argv


def main(argv=None):
    cli = parser()
    explicit = list(sys.argv[1:] if argv is None else argv)
    args = cli.parse_args(explicit)
    try:
        if args.makeup:
            allowed = {"--makeup", "--no-makeup", "--makeup-list", "--force-makeup", "--no-force-makeup",
                       "--dry-run", "--no-dry-run", "--progress-interval"}
            unsupported = [item.split("=", 1)[0] for item in explicit if item.startswith("--")
                           and item.split("=", 1)[0] not in allowed]
            if unsupported:
                raise ValueError(f"Makeup uses the original run's frozen protocol and the list selection; "
                                 f"these options are not supported: {', '.join(unsupported)}")
            from cbrs_quant.makeup import run_makeup
            from cbrs_quant.compact_sweep import compact_makeup
            import json
            document = json.loads(absolute(args.makeup_list).read_bytes())
            run = absolute(document["run"])
            if any(run.glob("*/model.json")):
                return compact_makeup(PROJECT_ROOT, absolute(args.makeup_list), dry_run=args.dry_run,
                                      force=args.force_makeup, interval=args.progress_interval)
            return run_makeup(PROJECT_ROOT, absolute(args.makeup_list), dry_run=args.dry_run,
                              force=args.force_makeup, interval=args.progress_interval)
        makeup_only = {"--makeup-list", "--force-makeup", "--no-force-makeup"}
        if any(item.split("=", 1)[0] in makeup_only for item in explicit):
            raise ValueError("Makeup options require --makeup; no regular sweep was started")
        sweep_args = sweep_arguments(args)
        result = run_sweep(sweep_args)
        if result == 0 and not args.dry_run:
            import subprocess
            run = Path(sweep_args[sweep_args.index("--output") + 1])
            report = run
            print(f"Experiments complete. Generating report: {report}", flush=True)
            return subprocess.call([sys.executable, "-B", str(PROJECT_ROOT / "analyze_results.py"),
                                    "--run", str(run), "--publish", str(report), "--update-readme"])
        return result
    except KeyboardInterrupt:
        message = ("Stopped. Finished makeup results are retained; run --makeup again to continue."
                   if args.makeup else "Stopped. Finished artifacts are retained; use --resume --output <run>.")
        print("\n" + message, flush=True)
        return 130
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
