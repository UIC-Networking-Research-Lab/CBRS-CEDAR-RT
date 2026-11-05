"""Rebuild the latest experiment report in results/; use --run to select another experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_DIR = None                         # Select the latest experiment under results/.
RUNS_ROOT = "results"
OUTPUT_ROOT = "results"                # Legacy analysis fallback only.
BASELINE = "torch_cuda_fp32"
FORMATS = ["png", "pdf"]
DPI = 180
SCOPE = "all"
LOW_PRIORITY = True                    # Best effort: yield CPU time to live experiments.

sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from cbrs_quant.analysis import collect, select_run, snapshot_directory, tables, write_report  # noqa: E402
from cbrs_quant.config import MODES  # noqa: E402


def absolute(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def lower_priority():
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000):
                raise ctypes.WinError(ctypes.get_last_error())  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    except OSError as exc:
        print(f"Note: could not lower analysis priority: {exc}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=RUN_DIR, help="Consolidated results directory or compact run")
    parser.add_argument("--runs-root", default=RUNS_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--baseline", choices=MODES, default=BASELINE)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=FORMATS)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--scope", choices=("all", "model_compute", "detector_pipeline", "image_to_detection"), default=SCOPE)
    parser.add_argument("--overview-scope", choices=("model_compute", "detector_pipeline", "image_to_detection"), default="model_compute")
    parser.add_argument("--publish", help="Write the complete portable report to this directory, e.g. results")
    parser.add_argument("--update-readme", action="store_true", help="Replace the root README Results section")
    args = parser.parse_args(argv)
    if args.dpi < 72 or not args.formats or len(set(args.formats)) != len(args.formats):
        parser.error("Use DPI >= 72 and unique output formats")
    try:
        if LOW_PRIORITY:
            lower_priority()
        from cbrs_quant.compact_analysis import newest_run
        if args.run:
            run = absolute(args.run)
        else:
            try:
                run = newest_run(absolute(args.runs_root))
            except FileNotFoundError:
                run = PROJECT_ROOT / "results"
        if not any(run.glob("*/model.json")) and ((run / "data/metadata.json").is_file() or (run / "metadata.json").is_file()):
            from cbrs_quant.results_store import data_directory
            from cbrs_quant.rt_report import generate, update_readme
            data = data_directory(run)
            output = absolute(args.publish) if args.publish else data.parent
            if output.is_relative_to(data) or (data.is_relative_to(output) and data != output / "data"):
                raise ValueError("Report output overlaps source data in an unsupported layout")
            print(f"Analysis input: {data}\nReport: {output}", flush=True)
            result = generate(data, output, tuple(args.formats), args.dpi, args.baseline, args.scope, args.overview_scope)
            if args.update_readme or args.run is None:
                update_readme(PROJECT_ROOT, output)
            print(f"Done: {result['models']} models, {result['comparisons']} comparison rows, "
                  f"{result['figures']} figures. Full report: {output / 'REPORT.md'}", flush=True)
            return 0
        public_manifest = run / "source_manifest.json"
        if (run / "source").is_dir() and public_manifest.is_file():
            import hashlib
            import json
            manifest = json.loads(public_manifest.read_text(encoding="utf-8"))
            for record in manifest["files"]:
                path = (run / "source" / record["path"]).resolve()
                if not path.is_relative_to((run / "source").resolve()):
                    raise ValueError("Published source path escapes its source directory")
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256_published"]:
                    raise ValueError(f"Published source hash mismatch: {record['path']}")
            run = run / "source"
        if any(run.glob("*/model.json")):
            from cbrs_quant.rt_report import generate, update_readme
            import json
            first = json.loads(next(run.glob("*/model.json")).read_text(encoding="utf-8"))
            name = first["experiment"].get("name", run.name)
            if Path(name).name != name or name in (".", "..") or "\\" in name:
                raise ValueError("Invalid experiment name")
            output = absolute(args.publish) if args.publish else run.resolve()
            if output != run.resolve() and (output.is_relative_to(run.resolve()) or run.resolve().is_relative_to(output)):
                raise ValueError("Use the experiment directory itself or a separate destination")
            dataset = run / "dataset.json"
            if not dataset.is_file():
                dataset = PROJECT_ROOT / "artifacts/dataset.json"
            print(f"Analysis input: {run}\nReport: {output}", flush=True)
            result = generate(run, output, tuple(args.formats), args.dpi, args.baseline, args.scope,
                              args.overview_scope, dataset if dataset.is_file() else None)
            if args.update_readme or args.run is None:
                update_readme(PROJECT_ROOT, output)
            print(f"Done: {result['models']} models, {result['comparisons']} comparison rows, "
                  f"{result['figures']} figure files.\nFull report: {output / 'REPORT.md'}\n"
                  "Experiment inputs were not changed.", flush=True)
            return 0
        if args.publish:
            raise ValueError("--publish requires a compact run or a published results bundle")
        # Existing legacy reports remain readable when explicitly selected.
        run = select_run(absolute(args.runs_root), run)
        print(f"Analysis input: {run}\nReading committed JSON reports only; no inference.", flush=True)
        snapshot = collect(run)
        quality, timings, trials = tables(snapshot, args.baseline)
        output = snapshot_directory(run, absolute(args.output_root), absolute(args.runs_root))
        # Tables remain useful even if plotting dependencies fail on another machine.
        write_report(snapshot, output, args.baseline, quality, timings, trials, [])
        print(f"Analysis output: {output}\nPlotting {len(snapshot['accepted'])} valid routes...", flush=True)
        # Keep font caches out of the project and experiment; remove them on exit.
        previous_cache = os.environ.get("MPLCONFIGDIR")
        try:
            with tempfile.TemporaryDirectory(prefix="cbrs-analysis-mpl-") as cache:
                os.environ["MPLCONFIGDIR"] = cache
                from cbrs_quant.analysis_plots import render
                figures = render(snapshot, quality, timings, output, args.baseline, tuple(args.formats), args.dpi)
        finally:
            if previous_cache is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_cache
        summary = write_report(snapshot, output, args.baseline, quality, timings, trials, figures)
        print(f"Done: {summary['valid_routes']}/{summary['planned_routes']} valid routes; "
              f"{len(figures)} figures.\nReport: {output / 'index.html'}", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
