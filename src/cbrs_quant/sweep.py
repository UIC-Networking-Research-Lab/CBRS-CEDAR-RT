"""Serial, resumable batch experiments with one child process per export/route.

Reports are committed after each phase. Failed attempts and model artifacts stay
on disk. A new attempt never overwrites a prior attempt. No batch is reduced.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import os
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .config import CPU_MODES, DEFAULT_BATCHES, DEFAULT_MODES, MODEL_IDS, MODES, execution_phases
from .io import digest, fingerprint, read_json, write_json
from .progress import ConsoleProgress

SCOPES = ("model_compute", "detector_pipeline", "image_to_detection")


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive_lock(path):
    """OS-owned advisory lock is released even if the supervisor crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
                stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def stage_state(attempt, **updates):
    path = attempt / "status.json"
    report = read_json(path) if path.exists() else {"started_at": now()}
    if updates.get("phase", report.get("phase")) != report.get("phase"):
        report["detail"] = {}
    report.update(updates, updated_at=now())
    write_json(path, report)


def worker_progress(attempt, phase):
    last = 0

    def update(detail):
        nonlocal last
        # Evaluation/calibration report at most once per second. Benchmark
        # callbacks only occur between measured loops, never inside their clocks.
        stamp = time.monotonic()
        if "total" in detail and detail.get("completed") != detail["total"] and stamp-last < 1:
            return
        stage_state(attempt, phase=phase, detail=detail)
        last = stamp
    return update


def artifact_hashes(attempt, paths):
    return {str(path.relative_to(attempt)): digest(path) for path in paths}


def valid_previous(result, job=None):
    """A success marker alone cannot certify missing or modified model/results."""
    artifacts = result.get("artifact_sha256", {})
    if not artifacts:
        return False
    root = Path(result["attempt"])
    try:
        if job is not None:
            old_job = read_json(root / "job.json")
            old_identity = {k: v for k, v in old_job.items() if k != "export_device"}
            new_identity = {k: v for k, v in job.items() if k != "export_device"}
            if old_identity != new_identity:
                return False
        return all(digest(root / path) == sha for path, sha in artifacts.items())
    except (OSError, ValueError):
        return False


def run_worker(job_path):
    job = read_json(job_path)
    # A worker may survive a killed supervisor. Its independent OS lock prevents
    # any resumed worker overlapping that orphan's measured GPU/CPU workload.
    with exclusive_lock(Path(job["output"]) / "worker.lock"):
        _run_worker(job_path)


def _run_worker(job_path):
    """The worker owns all CUDA objects; exiting releases the entire context."""
    import torch
    from .data import Dataset
    from .engine import build_engine
    from .export import export_onnx
    from .io import environment, sidecar
    from .runner import benchmark, evaluate, predict
    from .runtime import create_backend, is_cpu_mode, synchronize
    from .smoke import smoke

    job = read_json(job_path)
    attempt = job_path.parent
    torch.set_num_threads(job["threads"])
    stage_state(attempt, status="running", phase="starting", pid=os.getpid())
    try:
        weights = Path(job["weights"])
        if digest(weights) != job["weights_sha256"]:
            raise ValueError("Retained checkpoint hash changed")
        if digest(job["manifest"]) != job["manifest_sha256"]:
            raise ValueError("Frozen dataset manifest changed")
        write_json(attempt / "environment.json", environment())
        if job["kind"] == "export":
            dataset = Dataset(Path(job["data"]), Path(job["manifest"]), "train") if job["precision"] == "int8" else None
            graph = attempt / f"model.{job['precision']}.onnx"
            stage_state(attempt, phase="calibration_and_export")
            export_onnx(job["model_id"], weights, graph, batch=job["batch"], dataset=dataset,
                        samples=4 if job["smoke"] else 512,
                        precision="fp16" if job["precision"] == "fp16" else "fp32",
                        device_name=job["export_device"], progress=worker_progress(attempt, "calibration_and_export"))
            stage_state(attempt, status="passed", phase="complete", graph=str(graph),
                        graph_sha256=digest(graph), artifact_sha256=artifact_hashes(
                            attempt, [job_path, graph, sidecar(graph), attempt / "environment.json"]))
            return
        mode = job["mode"]
        dataset = Dataset(Path(job["data"]), Path(job["manifest"]), "val")
        if job["smoke"]:
            dataset.records = dataset.records[:8]
        paths = [dataset.path(r["image"]) for r in dataset.records[:3]]
        graph = Path(job["graph"]) if job.get("graph") else None
        engine = None
        level = 0 if job["smoke"] else 3
        if mode.startswith("trt_"):
            stage_state(attempt, phase="engine_build")
            engine = attempt / f"{mode}.engine"
            build_engine(graph, engine, precision=mode.removeprefix("trt_"), optimization_level=level)
        context = nullcontext() if is_cpu_mode(mode) else torch.cuda.stream(torch.cuda.Stream())
        with context:
            stage_state(attempt, phase="backend_load_and_provider_audit")
            started = time.perf_counter()
            backend = create_backend(mode, job["model_id"], weights, engine,
                                     job["batch"], graph, attempt / "ort_cache",
                                     threads=job["threads"], optimization_level=level)
            synchronize(backend)
            load_seconds = time.perf_counter() - started
            stage_state(attempt, phase="smoke_and_parity")
            parity = None if mode.endswith("int8") else weights
            write_json(attempt / "smoke.json", smoke(backend, paths, parity,
                       reference_device="cpu", reference_batch=1))
            if backend.device.type == "cuda":
                torch.cuda.empty_cache()
            predict(backend, paths, attempt / "demo")
            stage_state(attempt, phase="full_validation")
            quality = evaluate(backend, dataset, progress=worker_progress(attempt, "full_validation"))
            quality.update(smoke_only_subset=job["smoke"], load_seconds=load_seconds,
                           evaluated_images=len(dataset.records))
            write_json(attempt / "evaluation.json", quality)
            for scope in SCOPES:
                stage_state(attempt, phase=scope)
                timing = dict(warmup=2, min_images=4, seconds=.1, rounds=1, pool_size=4) if job["smoke"] else {}
                report = benchmark(backend, dataset, scope, progress=worker_progress(attempt, scope), **timing)
                report.update(smoke_only_subset=job["smoke"], load_seconds=load_seconds)
                write_json(attempt / f"{scope}.json", report)
        artifacts = [job_path, attempt / "environment.json", attempt / "smoke.json",
                     attempt / "evaluation.json", attempt / "demo" / "predictions.json"]
        artifacts.extend(attempt / f"{scope}.json" for scope in SCOPES)
        artifacts.extend((attempt / "demo").glob("*.png"))
        if engine:
            artifacts.extend([engine, sidecar(engine)])
        stage_state(attempt, status="passed", phase="complete", evaluated_images=len(dataset.records),
                    artifact_sha256=artifact_hashes(attempt, artifacts))
    except Exception as exc:
        stage_state(attempt, status="failed", error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc())
        raise


def job_plan(config, model_id, batch, weights, modes=None):
    common = {key: config[key] for key in ("data", "manifest", "manifest_sha256", "threads", "smoke", "output")}
    common.update(model_id=model_id, batch=batch, weights=str(weights),
                  weights_sha256=config["weights"][model_id]["sha256"])
    modes = config["modes"] if modes is None else modes
    for mode in modes:
        if mode.startswith("torch_"):
            yield mode, dict(common, kind="route", mode=mode), None
    precisions = set()
    for mode in modes:
        if not mode.startswith("torch_"):
            precisions.add("fp16" if mode == "ort_cuda_fp16" else "int8" if mode.endswith("int8") else "fp32")
    for precision in ("fp32", "fp16", "int8"):
        if precision in precisions:
            # Shared FP32 exports have identical identities in both phases.
            preference = "cpu" if all(m in CPU_MODES for m in config["modes"]) else "cuda"
            yield f"export_{precision}", dict(common, kind="export", precision=precision,
                                               export_preference=preference), None
    for mode in modes:
        if not mode.startswith("torch_"):
            precision = "fp16" if mode == "ort_cuda_fp16" else "int8" if mode.endswith("int8") else "fp32"
            yield mode, dict(common, kind="route", mode=mode), f"export_{precision}"


def attempt_result(stage):
    path = stage / "latest.json"
    return read_json(path) if path.exists() else None


def execute(config, stage, job, publish):
    import psutil
    previous = attempt_result(stage)
    if previous:
        if previous["status"] == "passed" and valid_previous(previous, job):
            return previous
        if previous["status"] in ("failed", "dependency_failed") and not config.get("retry_failed"):
            return previous
    # A fresh process permits CPU export after GPU OOM without a poisoned context.
    devices = ["cuda", "cpu"] if job["kind"] == "export" and job["precision"] != "fp16" else ["cuda"]
    if job.get("export_preference") == "cpu":
        devices = ["cpu"]
    for device in devices:
        if shutil.disk_usage(config["output"]).free < config["minimum_free_gib"] * 1024**3:
            raise OSError("Free disk space below sweep reserve; resume after freeing space")
        stage.mkdir(parents=True, exist_ok=True)
        number = 1 + max((int(p.name.split("_")[-1]) for p in stage.glob("attempt_*") if p.is_dir()), default=0)
        attempt = stage / f"attempt_{number:03d}"
        attempt.mkdir()
        current = dict(job, export_device=device)
        write_json(attempt / "job.json", current)
        env = os.environ.copy()
        env["PYTHONPATH"] = config["source"] + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"
        with (attempt / "worker.log").open("wb") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "cbrs_quant.sweep", "--worker",
                                        str(attempt / "job.json")], stdout=log, stderr=subprocess.STDOUT, env=env)
            started = time.monotonic()
            resource_error = None
            try:
                while True:
                    try:
                        state = read_json(attempt / "status.json")
                    except (OSError, ValueError):
                        state = {}
                    publish(active={"job": str(stage.relative_to(Path(config["output"]))),
                                    "attempt": str(attempt), "pid": process.pid,
                                    "model": job["model_id"], "batch": job["batch"],
                                    "mode": job.get("mode", f"export_{job.get('precision')}_{device}"),
                                    "state": state,
                                    "elapsed_seconds": time.monotonic()-started})
                    try:
                        code = process.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        try:
                            rss = psutil.Process(process.pid).memory_info().rss
                            available = psutil.virtual_memory().available
                            if (rss > config["maximum_worker_gib"] * 1024**3
                                    or (available < 512 * 1024**2 and rss > 2 * 1024**3)):
                                resource_error = "Host memory guard reached; requested batch was not reduced"
                                process.terminate()
                                code = process.wait()
                                break
                        except psutil.NoSuchProcess:
                            pass
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                stage_state(attempt, status="interrupted")
                raise
        status_file = attempt / "status.json"
        result = read_json(status_file) if status_file.exists() else {}
        if code != 0 or result.get("status") != "passed":
            result.update(status="failed", returncode=code)
            result.setdefault("error", "Worker exited before committing a success report; see worker.log")
        if resource_error:
            result.update(status="failed", error=resource_error, resource_limit="host_memory")
        result.update(attempt=str(attempt), elapsed_seconds=time.monotonic()-started)
        write_json(status_file, result)
        write_json(stage / "latest.json", result)
        if result["status"] == "passed":
            break
        # FP32/QDQ fallback changes export hardware only, never the inference mode.
    return result


def summarize_group(group, modes):
    from .compare import compare
    result = {mode: attempt_result(group / "results" / mode) for mode in modes}
    for scope in SCOPES:
        paths = [Path(r["attempt"]) / f"{scope}.json" for r in result.values()
                 if r and r["status"] == "passed"]
        if paths:
            compare(paths, group / "comparison" / scope)
    write_json(group / "experiment.json", result)


def supervise(config):
    root = Path(config["output"])
    status = {"status": "running", "supervisor_pid": os.getpid(), "started_at": now(),
              "planned_routes": len(config["weights"])*len(config["batches"])*len(config["modes"]),
              "completed_groups": 0, "counts": {}}
    results = {}
    completed_groups = set()
    console = ConsoleProgress(config.get("progress_interval", 10), config.get("console", True))

    def publish(**updates):
        status.update(updates, updated_at=now(), free_disk_bytes=shutil.disk_usage(root).free)
        status["counts"] = dict(Counter(results.values()))
        write_json(root / "progress.json", status)
        console(status)

    with exclusive_lock(root / "supervisor.lock"):
        # Refuse resume while an orphaned worker still owns the device. Never
        # kill an unverified PID or treat a released supervisor lock as idle GPU.
        with exclusive_lock(root / "worker.lock"):
            pass
        try:
            for phase, modes in execution_phases(config["modes"]):
                publish(phase=phase, active=None, last_result=None)
                for batch, model_id in group_order(config):
                    source = config["weights"][model_id]
                    group = root / f"batch_{batch:03d}" / model_id
                    group.mkdir(parents=True, exist_ok=True)
                    weights = group / "models" / "checkpoint.pt"
                    weights.parent.mkdir(exist_ok=True)
                    if not weights.exists():
                        try:
                            os.link(source["retained_path"], weights)
                        except OSError:
                            shutil.copyfile(source["retained_path"], weights)
                    if digest(weights) != source["sha256"]:
                        raise ValueError(f"Retained checkpoint hash mismatch: {weights}")
                    jobs = {}
                    for name, job, dependency in job_plan(config, model_id, batch, weights, modes):
                        stage = group / ("models" if job["kind"] == "export" else "results") / name
                        if dependency and jobs[dependency]["status"] != "passed":
                            result = {"status": "dependency_failed", "dependency": dependency,
                                      "error": "Required ONNX export failed; inference not executed"}
                            write_json(stage / "latest.json", result)
                        else:
                            if dependency:
                                job["graph"] = jobs[dependency]["graph"]
                            result = execute(config, stage, job, publish)
                        jobs[name] = result
                        if job["kind"] == "route":
                            results[f"{batch}/{model_id}/{name}"] = result["status"]
                        publish(active=None, last_result={
                            "job": str(stage), "model": model_id, "batch": batch, "mode": name,
                            "status": result["status"], "error": result.get("error"), "directory": str(stage)})
                    # A resumed CPU result may be stale; do not read it until
                    # its phase has verified/repaired the corresponding stage.
                    checked_modes = [m for m in config["modes"] if f"{batch}/{model_id}/{m}" in results]
                    summarize_group(group, checked_modes)
                    if all(f"{batch}/{model_id}/{mode}" in results for mode in config["modes"]):
                        completed_groups.add((batch, model_id))
                    status["completed_groups"] = len(completed_groups)
            publish(status="completed" if all(s == "passed" for s in results.values()) else "completed_with_failures",
                    active=None, last_result=None)
            print(f"Finished: {dict(Counter(results.values()))}. Results: {root}", flush=True)
            return status
        except BaseException as exc:
            publish(status="interrupted", error=f"{type(exc).__name__}: {exc}", active=None, last_result=None)
            raise


def group_order(config):
    return ((batch, model_id) for batch in config["batches"] for model_id in config["model_order"])


def create_config(args):
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError("Use --resume to continue an existing sweep")
    assets = read_json(args.weights_manifest)
    lookup = {record["model_id"]: record for record in assets}
    requested = args.models or list(lookup)
    if not requested or set(requested)-set(lookup):
        raise ValueError("Requested models are not all present in the weights manifest")
    selected = {model_id: lookup[model_id] for model_id in requested}
    # Freeze runnable sources and inputs before starting a multi-day job.
    source = root / "source"
    shutil.copytree(Path(__file__).parent, source / "cbrs_quant",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = root / "dataset.json"
    shutil.copyfile(args.manifest, manifest)
    for model_id, entry in selected.items():
        original = (args.project_root / entry["path"]).resolve()
        if digest(original) != entry["sha256"]:
            raise ValueError(f"Checkpoint manifest mismatch: {model_id}")
        target = root / "checkpoints" / f"{model_id}.pt"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, target)
        if digest(target) != entry["sha256"]:
            raise ValueError(f"Checkpoint retention failed: {model_id}")
        entry["retained_path"] = str(target)
    config = {"schema": 1, "created_at": now(), "output": str(root), "source": str(source),
              "data": str(args.data.resolve()), "manifest": str(manifest),
              "manifest_sha256": digest(manifest), "weights": selected,
              "model_order": list(selected),
              "batches": args.batches, "modes": args.modes, "threads": args.threads,
              "execution_order": "all_gpu_then_all_cpu", "progress_interval": args.progress_interval,
              "smoke": args.smoke, "minimum_free_gib": args.minimum_free_gib,
              "maximum_worker_gib": args.maximum_worker_gib,
              "protocol": {"seconds": .1 if args.smoke else 30, "rounds": 1 if args.smoke else 3,
                           "calibration_images": 4 if args.smoke else 512,
                           "evaluation_split": "val", "scopes": list(SCOPES)},
              "source_hashes": {p.relative_to(source).as_posix(): digest(p) for p in source.rglob("*.py")}}
    config["config_sha256"] = fingerprint(config)
    write_json(root / "config.json", config)
    return config


def planned_models(args):
    selected = args.models or ["all"]
    if len(set(selected)) != len(selected) or ("all" in selected and selected != ["all"]):
        raise ValueError("Use unique model IDs or just 'all'")
    unknown = set(selected) - set(MODEL_IDS) - {"all"}
    if unknown:
        raise ValueError(f"Unsupported models: {sorted(unknown)}")
    if args.weights_manifest.is_file():
        assets = read_json(args.weights_manifest)
        available = [record["model_id"] for record in assets]
        if len(set(available)) != len(available) or not available or set(available)-set(MODEL_IDS):
            raise ValueError("Weight manifest must contain unique supported model IDs")
        if selected == ["all"]:
            return available
        if set(selected)-set(available):
            raise ValueError("Selected model is missing from the weight manifest")
    return list(MODEL_IDS) if selected == ["all"] else selected


def print_plan(plan):
    models, modes, batches = plan["model_order"], plan["modes"], plan["batches"]
    groups = len(models)*len(batches)
    print("\nCBRS experiment plan", flush=True)
    print(f"  Models ({len(models)}): {', '.join(models)}", flush=True)
    print(f"  Batches: {', '.join(map(str, batches))} | CPU threads: {plan['threads']}", flush=True)
    phases = execution_phases(modes)
    for phase, subset in phases:
        print(f"  {phase}: {groups*len(subset)} routes | {', '.join(subset)}", flush=True)
    order = " -> ".join(phase for phase, _ in phases)
    print(f"  Total: {groups*len(modes)} routes; order: {order} (all models/batches per phase).", flush=True)
    calibration = str(4 if plan["smoke"] else 512) + " PNGs" if any(m.endswith("int8") for m in modes) else "none (no INT8)"
    if plan["smoke"]:
        print(f"  Smoke: calibration: {calibration} / up to 8 validation PNGs; short timings.", flush=True)
    else:
        hours = groups*len(modes)*len(SCOPES)*3*2*30/3600
        print(f"  Formal: calibration: {calibration} / full val; 3 scopes, 3 rounds, 50 warmups.", flush=True)
        print(f"  Each latency/throughput loop: >=30s AND >=1000 images. Timing alone >= {hours:g}h.", flush=True)
    print(f"  PNGs: {plan['data']}\n  Output: {plan['output']}", flush=True)
    print("  Live progress is printed below; full native logs stay in each attempt/worker.log.", flush=True)


def preflight(plan, weights_manifest=None):
    """Fail before copying assets/starting workers; never install dependencies."""
    errors = []
    for label, path in (("PNG dataset", plan["data"]), ("dataset manifest", plan["manifest"]),
                        ("weights manifest", weights_manifest)):
        if path is not None and not Path(path).exists():
            errors.append(f"Missing {label}: {path}")
    modules = {"numpy", "PIL", "yaml", "torch", "torchvision", "psutil"}
    if any(not m.startswith("torch_") for m in plan["modes"]):
        modules.add("onnx")
    if any(m.startswith("ort_") for m in plan["modes"]):
        modules.add("onnxruntime")
    if any(m.startswith(("trt_", "ort_trt_")) for m in plan["modes"]):
        modules.add("tensorrt")
    if any(m.startswith("cbrs_yolo") for m in plan["model_order"]):
        modules.add("ultralytics")
    missing = [name for name in sorted(modules) if importlib.util.find_spec(name) is None]
    if missing:
        errors.append(f"Missing Python modules in {sys.executable}: {', '.join(missing)}")
    if errors:
        raise RuntimeError("Cannot start experiments:\n  " + "\n  ".join(errors) +
                           "\nPrepare dependencies and inputs as described in README.md; --dry-run needs neither.")


def validate_resume(config, output, current_source=None):
    if Path(config["output"]).resolve() != output.resolve():
        raise ValueError("Run was moved: resume at its original path, or start a new run")
    if fingerprint({k: v for k, v in config.items() if k != "config_sha256"}) != config["config_sha256"]:
        raise ValueError("Sweep configuration changed")
    current_source = current_source or Path(__file__).resolve().parents[1]
    for relative, sha in config["source_hashes"].items():
        if digest(Path(config["source"]) / relative) != sha:
            raise ValueError("Frozen experiment source changed")
        active = current_source / relative
        if not active.is_file() or digest(active) != sha:
            raise ValueError("Current code differs from the frozen run. Resume with PYTHONPATH set to "
                             f"{config['source']} and python -m cbrs_quant.sweep --resume --output {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--weights-manifest", type=Path, default=Path("artifacts/weights/manifest.json"))
    parser.add_argument("--data", type=Path, default=Path("data/yolo_split"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/dataset.json"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=DEFAULT_MODES)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--minimum-free-gib", type=float, default=6)
    parser.add_argument("--maximum-worker-gib", type=float, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without imports, writes or workers")
    parser.add_argument("--progress-interval", type=float, default=10)
    args = parser.parse_args(argv)
    if args.worker:
        run_worker(args.worker)
        return
    if args.output is None:
        parser.error("--output is required")
    if (min(args.batches) < 1 or args.threads < 1 or args.minimum_free_gib < 0
            or args.maximum_worker_gib <= 0 or args.progress_interval <= 0):
        parser.error("Positive batches/threads and nonnegative disk reserve are required")
    if len(set(args.batches)) != len(args.batches) or len(set(args.modes)) != len(args.modes):
        parser.error("Duplicate batches/modes are not allowed")
    if args.resume:
        config = read_json(args.output / "config.json")
        validate_resume(config, args.output)
        print("Resume: using the saved models, batches, modes and timing protocol.", flush=True)
        print_plan(config)
        if args.dry_run:
            print("Plan only: no files written and no workers started.", flush=True)
            return 0
        preflight(config)
    else:
        args.models = planned_models(args)
        plan = {**vars(args), "model_order": args.models}
        print_plan(plan)
        if args.dry_run:
            print("Plan only: no files written and no workers started.", flush=True)
            return 0
        preflight(plan, args.weights_manifest)
        print("Preparing retained checkpoints and source snapshot...", flush=True)
        config = create_config(args)
    config["retry_failed"] = args.retry_failed
    config["progress_interval"] = args.progress_interval
    result = supervise(config)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
