"""Serial GPU-first sweep writing one compact directory per model."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time

from .compact import EXPORT_FILES, copy_checkpoint, current_signature, graph_name, initialize_model, put_route, read_csv
from .config import CPU_MODES, DEFAULT_BATCHES, DEFAULT_MODES, MODES
from .io import digest, read_json, write_json, verified_artifact, onnx_precision
from .provenance import checkpoint_info, experiment_info
from . import sweep


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def execution_lock(root):
    """Prevent concurrent compact sweeps in the same output root."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.CreateMutexW.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        name = "Local\\CBRSQuant_" + hashlib.sha256(str(root).lower().encode()).hexdigest()
        handle = kernel.CreateMutexW(None, False, name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = kernel.WaitForSingleObject(handle, 0) in (0, 0x80)
        try:
            if not acquired:
                raise RuntimeError(f"Another experiment is active under {root}")
            yield
        finally:
            if acquired:
                kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)
    else:
        import fcntl
        fd = os.open(root, os.O_RDONLY)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another experiment is active under {root}") from exc
            yield
        finally:
            os.close(fd)


def print_plan(config):
    models, batches, modes = config["model_order"], config["batches"], config["modes"]
    groups = len(models) * len(batches)
    print(f"\nCBRS experiment plan\n  Models ({len(models)}): {', '.join(models)}", flush=True)
    print(f"  Batches: {', '.join(map(str, batches))} | CPU threads: {config['threads']}", flush=True)
    for phase, selected in sweep.execution_phases(modes):
        print(f"  {phase}: {groups * len(selected)} routes | {', '.join(selected)}", flush=True)
    print(f"  Total: {groups * len(modes)} routes; order: GPU -> CPU.", flush=True)
    if config["smoke"]:
        print("  Smoke: short timings, up to 8 validation PNGs.", flush=True)
    else:
        hours = groups * len(modes) * len(sweep.SCOPES) * 3 * 2 * 30 / 3600
        print(f"  Formal: 512 calibration PNGs, full validation, 3 scopes, 3 rounds; timing >= {hours:g}h.", flush=True)
    print(f"  PNGs: {config['data']}\n  Output: {config['output']}", flush=True)
    print("  Each model has results.csv, model.json, comparison.csv, demo/ and ONNX files.", flush=True)


def execute(task, maximum_worker_gib, interval=10):
    """A process per route releases GPU memory; temporary runtime files stay outside output/."""
    import psutil
    from .compact_worker import run
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="cbrs-runtime-") as folder:
        task = {**task, "scratch": folder}
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=run, args=(task, send), daemon=True)
        started, last_print, current = time.monotonic(), 0, "starting"
        process.start()
        send.close()
        result = None
        try:
            try:
                while process.is_alive() or receive.poll():
                    if receive.poll(1):
                        try:
                            event = receive.recv()
                        except EOFError:
                            break
                        if event.get("type") == "result":
                            result = event
                            break
                        if event.get("type") == "progress":
                            phase, detail = event.get("phase"), event.get("detail") or {}
                            if phase:
                                current = phase
                            if "total" in detail:
                                current = f"{current} {detail.get('completed', 0)}/{detail['total']} images"
                            elif "round" in detail:
                                current = f"{phase or current} {detail['round']}/{detail['rounds']} {detail.get('measurement', '')}"
                            elif "measurement" in detail:
                                current = f"{phase or current} {detail['measurement']}"
                    stamp = time.monotonic()
                    if stamp-last_print >= interval:
                        print(f"    {current} | {int(stamp-started)}s", flush=True)
                        last_print = stamp
                    try:
                        rss = psutil.Process(process.pid).memory_info().rss
                        available = psutil.virtual_memory().available
                        if rss > maximum_worker_gib * 1024**3 or (available < 512 * 1024**2 and rss > 2 * 1024**3):
                            process.terminate()
                            result = {"state": {"status": "failed", "phase": current,
                                                "error": "Host memory guard reached"}, "reports": {}}
                            break
                    except psutil.NoSuchProcess:
                        pass
                process.join(timeout=5)
            except KeyboardInterrupt:
                result = {"state": {"status": "interrupted", "phase": current, "error": "Interrupted by user"},
                          "reports": {}, "interrupted": True}
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            receive.close()
        if result is None:
            result = {"state": {"status": "failed", "phase": current,
                                "error": f"Worker exited without a result (exit code {process.exitcode})"},
                      "reports": {}}
        # A stopped worker may have finished validation or some timing scopes.
        for name in ("evaluation", "smoke", *sweep.SCOPES):
            path = Path(folder) / "attempt" / f"{name}.json"
            if path.is_file() and name not in result.get("reports", {}):
                try:
                    result.setdefault("reports", {})[name] = read_json(path)
                except (OSError, ValueError):
                    pass
        result["elapsed_seconds"] = time.monotonic()-started
        return result


def plan_entries(config):
    return [{"model": model, "batch": batch, "mode": mode}
            for cpu in (False, True) for batch in config["batches"] for model in config["model_order"]
            for mode in config["modes"] if (mode in CPU_MODES) == cpu]


def original_checkpoint(project, record):
    path = Path(record["path"])
    return (path if path.is_absolute() else project / path).resolve()


def project_path(project, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project / path).resolve()


def portable_path(project, value):
    path = Path(value).resolve()
    project = Path(project).resolve()
    return path.relative_to(project).as_posix() if path.is_relative_to(project) else str(path)


def initialize(args, models):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Output already exists; use --resume for an existing compact run")
    assets = {row["model_id"]: row for row in read_json(args.weights_manifest)}
    dataset = read_json(args.manifest)
    checkpoints = {}
    for model in models:
        checkpoint = original_checkpoint(args.project_root, assets[model])
        if digest(checkpoint) != assets[model]["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {model}")
        checkpoints[model] = (checkpoint, checkpoint_info(checkpoint))
    config = {"created_at": now(), "output": str(output), "model_order": models, "batches": args.batches,
              "modes": args.modes, "data": str(args.data.resolve()), "manifest": str(args.manifest.resolve()),
              "manifest_sha256": digest(args.manifest), "threads": args.threads, "smoke": args.smoke,
              "minimum_free_gib": args.minimum_free_gib, "maximum_worker_gib": args.maximum_worker_gib,
              "protocol": {"seconds": .1 if args.smoke else 30, "rounds": 1 if args.smoke else 3,
                           "calibration_images": 4 if args.smoke else 512, "evaluation_split": "val",
                           "scopes": list(sweep.SCOPES)}}
    output.mkdir(parents=True)
    plan = experiment_info(config, output.name, None)
    for field in ("data", "manifest"):
        plan[field] = portable_path(args.project_root, plan[field])
    for model in models:
        checkpoint, storage = checkpoints[model]
        folder = initialize_model(output, model, plan, dataset, assets[model]["sha256"])
        meta = read_json(folder / "model.json")
        # FP32 export starts from checkpoint values; casting cannot restore lost precision.
        meta["checkpoint_storage"] = storage
        write_json(folder / "model.json", meta)
        copy_checkpoint(folder, checkpoint)
    return config


def read_config(output, project=None):
    paths = sorted(output.glob("*/model.json"))
    if not paths:
        raise FileNotFoundError(f"No compact model results in {output}")
    meta = read_json(paths[0])
    plan = meta["experiment"]
    if Path(plan.get("output", output)).resolve() != output.resolve():
        raise ValueError("Compact run moved; use its original data and checkpoint paths")
    project = Path(project or Path(__file__).resolve().parents[2])
    return {**plan, "output": str(output),
            "data": str(project_path(project, plan["data"])),
            "manifest": str(project_path(project, plan["manifest"]))}


def compact_makeup(project, list_path, *, dry_run=False, force=False, interval=10):
    raw = list_path.read_bytes()
    list_sha = hashlib.sha256(raw).hexdigest()
    document = json.loads(raw)
    if document.get("schema") != 1 or not isinstance(document.get("experiments"), list):
        raise ValueError("Invalid makeup list")
    path = Path(document["run"]).expanduser()
    output = (path if path.is_absolute() else project / path).resolve()
    config = read_config(output, project)
    selected, seen = [], set()
    for row in document["experiments"]:
        model, batch, mode = row.get("model"), row.get("batch"), row.get("mode")
        key = (model, batch, mode)
        if (model not in config["model_order"] or type(batch) is not int or batch not in config["batches"]
                or mode not in config["modes"] or key in seen):
            raise ValueError(f"Invalid or duplicate makeup route: {row}")
        seen.add(key)
        selected.append({"model": model, "batch": batch, "mode": mode})
    if not selected:
        raise ValueError("Makeup list is empty")
    selected.sort(key=lambda row: row["mode"] in CPU_MODES)
    print(f"\nMakeup: {len(selected)} routes; output: {output}", flush=True)
    for i, item in enumerate(selected, 1):
        print(f"  {i:>2}. {item['model']} b={item['batch']} {item['mode']}", flush=True)
    if dry_run:
        print("Plan only: no files written and no workers started.", flush=True)
        return 0
    with execution_lock(output.parent):
        sweep.preflight(config)
        return run_routes(output, config, selected, interval, makeup=True, list_sha=list_sha, force=force)


def run_routes(output, config, entries, interval, *, makeup=False, list_sha=None, force=False,
               retry_failed=False):
    output = output.resolve()
    code_hash = current_signature()
    export_hash = current_signature(EXPORT_FILES)
    results = []
    for index, entry in enumerate(entries, 1):
        model, batch, mode = entry["model"], entry["batch"], entry["mode"]
        model_dir = output / model
        meta = read_json(model_dir / "model.json")
        previous = meta.get("routes", {}).get(f"{batch}:{mode}", {})
        if makeup:
            skip = (not force and previous.get("status") == "passed"
                    and previous.get("execution_kind") == "makeup" and previous.get("makeup_list_sha256") == list_sha)
        else:
            skip = previous.get("status") == "passed" or (previous.get("status") in ("failed", "dependency_failed") and not retry_failed)
        if skip:
            print(f"[{index}/{len(entries)}] {model} b={batch} {mode} | existing {previous['status']}", flush=True)
            results.append(previous["status"])
            continue
        if shutil.disk_usage(output).free < config.get("minimum_free_gib", 0) * 1024**3:
            raise OSError("Free disk space is below the configured reserve")
        phase = "CPU" if mode in CPU_MODES else "GPU"
        print(f"[{index}/{len(entries)} | {phase}] {model} b={batch} {mode}", flush=True)
        started = now()
        checkpoint = Path(meta["checkpoint_path"])
        if not checkpoint.is_absolute():
            checkpoint = model_dir / checkpoint
        common = {"model_id": model, "batch": batch, "weights": str(checkpoint),
                  "weights_sha256": meta["checkpoint_sha256"], "data": config["data"],
                  "manifest": config["manifest"], "manifest_sha256": config["manifest_sha256"],
                  "threads": config["threads"], "smoke": config["smoke"]}
        graph = None
        if not mode.startswith("torch_"):
            precision = "fp16" if mode == "ort_cuda_fp16" else "int8" if mode.endswith("int8") else "fp32"
            graph = model_dir / graph_name(batch, precision)
            try:
                graph_meta = verified_artifact(graph, "onnx")
                expected_precision = "int8_qdq" if precision == "int8" else precision
                valid_graph = (graph_meta.get("batch") == batch and graph_meta.get("model_id") == model
                               and graph_meta.get("weights_sha256") == meta["checkpoint_sha256"]
                               and onnx_precision(graph_meta) == expected_precision
                               and meta.get("onnx", {}).get(f"{batch}:{precision}", {}).get("export_code_sha256") == export_hash)
            except (OSError, ValueError, KeyError, TypeError):
                valid_graph = False
            if not valid_graph:
                devices = ("cuda",) if precision == "fp16" else (("cpu",) if all(m in CPU_MODES for m in config["modes"]) else ("cuda", "cpu"))
                exported = None
                for device in devices:
                    job = {**common, "kind": "export", "precision": precision, "export_device": device}
                    exported = execute({"job": job, "model_dir": str(model_dir)}, config["maximum_worker_gib"], interval)
                    if exported.get("interrupted") or exported["state"].get("status") == "passed":
                        break
                if exported["state"].get("status") != "passed":
                    state = {"status": "interrupted" if exported.get("interrupted") else "dependency_failed", "phase": "export",
                             "error": exported["state"].get("error", "ONNX export failed")}
                    info = {**state, "started_at": started, "finished_at": now(), "inference_code_sha256": code_hash,
                            "execution_kind": "makeup" if makeup else "original", "makeup_list_sha256": list_sha}
                    put_route(model_dir, batch, mode, state["status"], {}, info)
                    results.append(state["status"])
                    print(f"    {state['status']}: {state['error']}", flush=True)
                    if exported.get("interrupted"):
                        raise KeyboardInterrupt
                    continue
        job = {**common, "kind": "route", "mode": mode, "graph": str(graph) if graph else None}
        capture_demo = len(list((model_dir / "demo").glob("*.png"))) < 10
        outcome = execute({"job": job, "model_dir": str(model_dir), "capture_demo": capture_demo},
                          config["maximum_worker_gib"], interval)
        state = outcome["state"]
        status = state.get("status", "failed")
        info = {"started_at": started, "finished_at": now(), "elapsed_seconds": outcome["elapsed_seconds"],
                "inference_code_sha256": code_hash,
                "phase": state.get("phase", ""), "error": state.get("error", ""),
                "execution_kind": "makeup" if makeup else "original", "makeup_list_sha256": list_sha}
        if state.get("traceback"):
            info["traceback"] = state["traceback"]
        if state.get("demo"):
            model_meta = read_json(model_dir / "model.json")
            model_meta["demo"] = state["demo"]
            write_json(model_dir / "model.json", model_meta)
        put_route(model_dir, batch, mode, status, outcome.get("reports", {}), info)
        results.append(status)
        print(f"    {status.upper()}" + (f": {info['error']}" if info["error"] else ""), flush=True)
        if outcome.get("interrupted"):
            raise KeyboardInterrupt
    print(f"Finished: {dict(Counter(results))}. Output: {output}", flush=True)
    return 0 if all(status == "passed" for status in results) else 1


def parser():
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--manifest", type=Path, required=True)
    cli.add_argument("--weights-manifest", type=Path, required=True)
    cli.add_argument("--models", nargs="+", default=["all"])
    cli.add_argument("--batches", nargs="+", type=int, default=list(DEFAULT_BATCHES))
    cli.add_argument("--modes", nargs="+", choices=MODES, default=DEFAULT_MODES)
    cli.add_argument("--threads", type=int, default=1)
    cli.add_argument("--progress-interval", type=float, default=10)
    cli.add_argument("--minimum-free-gib", type=float, default=6)
    cli.add_argument("--maximum-worker-gib", type=float, default=8)
    for name in ("smoke", "dry-run", "resume", "retry-failed"):
        cli.add_argument("--" + name, action="store_true")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if (not args.batches or min(args.batches) < 1 or len(set(args.batches)) != len(args.batches)
            or len(set(args.modes)) != len(args.modes) or args.threads < 1 or args.progress_interval <= 0
            or args.minimum_free_gib < 0 or args.maximum_worker_gib <= 0):
        raise ValueError("Invalid batches, modes or resource limits")
    if args.resume:
        config = read_config(args.output, args.project_root)
        print_plan(config)
        if args.dry_run:
            return 0
        with execution_lock(args.output.resolve().parent):
            sweep.preflight(config)
            return run_routes(args.output, config, plan_entries(config), args.progress_interval,
                              retry_failed=args.retry_failed)
    else:
        models = sweep.planned_models(args)
        plan = {**vars(args), "model_order": models}
        print_plan(plan)
        if args.dry_run:
            print("Plan only: no files written and no workers started.", flush=True)
            return 0
        with execution_lock(args.output.resolve().parent):
            sweep.preflight(plan, args.weights_manifest)
            config = initialize(args, models)
            return run_routes(args.output, config, plan_entries(config), args.progress_interval,
                              retry_failed=args.retry_failed)
