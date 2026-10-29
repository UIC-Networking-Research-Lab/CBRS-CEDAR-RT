"""Rerun an explicit list inside an existing sweep, preserving prior attempts."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .analysis import ENVIRONMENT_FIELDS, attempt_name, read_bytes
from .config import CPU_MODES
from .io import digest, write_json
from .progress import ConsoleProgress
from . import sweep


def now():
    return datetime.now(timezone.utc).isoformat()


def load_plan(path: Path, project: Path):
    raw = path.read_bytes()
    list_sha = hashlib.sha256(raw).hexdigest()
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("schema") != 1:
        raise ValueError("Makeup list must be a schema 1 JSON object")
    if not isinstance(document.get("run"), str) or not document["run"].strip():
        raise ValueError("Makeup list must specify the original run directory")
    run = Path(document["run"]).expanduser()
    run = (run if run.is_absolute() else project / run).resolve()
    config = json.loads(read_bytes(run / "config.json", run.parent))
    within_run(run, config["source"])
    within_run(run, config["manifest"])
    sweep.validate_resume(config, run, project / "src")
    entries = document.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Makeup list must contain at least one experiment")
    selected, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every experiment must be an object with model, batch and mode")
        model, batch, mode = entry.get("model"), entry.get("batch"), entry.get("mode")
        if (not isinstance(model, str) or model not in config["model_order"]
                or type(batch) is not int or batch not in config["batches"]
                or not isinstance(mode, str) or mode not in config["modes"]):
            raise ValueError(f"Experiment is outside the original run matrix: {entry}")
        key = (model, batch, mode)
        if key in seen:
            raise ValueError(f"Duplicate experiment: {key}")
        seen.add(key)
        selected.append({"model": model, "batch": batch, "mode": mode,
                         "reason": str(entry.get("reason", "user_selected"))})
    # Preserve the list order within each phase while prioritizing GPU routes.
    selected.sort(key=lambda row: row["mode"] in CPU_MODES)
    return run, config, selected, list_sha


def print_plan(run, config, entries, list_path, force):
    gpu = sum(row["mode"] not in CPU_MODES for row in entries)
    print(f"\nMakeup plan: {len(entries)} routes ({gpu} GPU, {len(entries)-gpu} CPU)", flush=True)
    print(f"  Original run: {run}\n  List: {list_path}", flush=True)
    print(f"  Protocol: {'smoke' if config['smoke'] else 'formal'}; "
          f"{config['protocol']['rounds']} rounds; results stay in their original model/batch folders.", flush=True)
    protocol = config["protocol"]
    minimum_minutes = len(entries) * len(protocol["scopes"]) * protocol["rounds"] * 2 * protocol.get("seconds", 0) / 60
    if not config["smoke"]:
        print(f"  For all selected routes to complete: timing loops >= {minimum_minutes:g} minutes, "
              "plus export, calibration and validation; early failures can finish sooner.", flush=True)
    print(f"  Memory guard: worker > {config['maximum_worker_gib']:g} GiB, or system available "
          "< 512 MiB while worker > 2 GiB. A resource failure may recur.", flush=True)
    print("  Completed routes from this exact list " + ("will be repeated." if force else "will be reused."), flush=True)
    for number, item in enumerate(entries, 1):
        print(f"  {number:>2}. batch={item['batch']} {item['model']} {item['mode']} | {item['reason']}", flush=True)


def within_run(run, path):
    target = Path(path).resolve()
    if not target.is_relative_to(run):
        raise ValueError(f"Result path escapes the original run: {target}")
    return target


def validate_environment(project, run, config, entries):
    """A short subprocess avoids retaining CUDA state in the long-lived supervisor."""
    import subprocess
    import sys
    code = ("import json,sys; sys.path.insert(0, sys.argv[1]); "
            "from cbrs_quant.io import environment; import torch; "
            "torch.set_num_threads(int(sys.argv[2])); print(json.dumps(environment()))")
    result = subprocess.run([sys.executable, "-E", "-B", "-c", code, str(project / "src"),
                             str(config["threads"])], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Environment check failed in {sys.executable}: {result.stderr.strip()}")
    current = json.loads(result.stdout)
    if any(item["mode"] not in CPU_MODES for item in entries) and not current.get("cuda_available"):
        raise RuntimeError("GPU makeup routes require CUDA in the selected Python environment")
    # Compare each selected group with a prior completed route in that group;
    # a group with no completed route uses an existing baseline from the run.
    def reference(stages, statuses=("passed",)):
        for stage in stages:
            stage = within_run(run, stage)
            latest_path = stage / "latest.json"
            if not latest_path.is_file():
                continue
            latest = json.loads(read_bytes(latest_path, run.parent))
            if latest.get("status") in statuses and latest.get("attempt"):
                attempt = within_run(run, stage / attempt_name(latest["attempt"]))
                path = attempt / "environment.json"
                if path.is_file():
                    return json.loads(read_bytes(path, run.parent))
        return None
    all_stages = sorted(run.glob("batch_*/*/results/*"))
    fallback = reference(all_stages) or reference(all_stages, ("failed",))
    if fallback is None:
        raise RuntimeError("No recorded reference environment found in the original run")
    for model, batch in dict.fromkeys((item["model"], item["batch"]) for item in entries):
        group = within_run(run, run / f"batch_{batch:03d}" / model / "results")
        group_stages = sorted(group.glob("*"))
        baseline = reference(group_stages) or reference(group_stages, ("failed",)) or fallback
        mismatch = [key for key in ENVIRONMENT_FIELDS if current.get(key) != baseline.get(key)]
        if mismatch:
            raise RuntimeError(f"Environment differs for {model} batch {batch}: {', '.join(mismatch)}. "
                               "Use the original environment; existing results were not changed.")


@contextmanager
def safe_sweep_io(run, stage=None, metadata=None, force_stage=None):
    """Keep frozen workers untouched; use safe Windows reads and tag committed retries."""
    original_read, original_write, original_attempt = sweep.read_json, sweep.write_json, sweep.attempt_result

    def safe_read(path):
        path = Path(path).resolve()
        if not path.is_relative_to(run):
            return original_read(path)
        return json.loads(read_bytes(path, run.parent))

    def tagged_write(path, value):
        target = Path(path).resolve()
        if (metadata is not None and target.name == "job.json"
                and target.is_relative_to(stage.resolve())):
            original_write(path, value)
            original_write(target.parent / "makeup.json", metadata)
            return
        if (metadata is not None and target.name in ("status.json", "latest.json")
                and target.is_relative_to(stage.resolve()) and isinstance(value, dict)):
            value.update(metadata, makeup_finished_at=now())
            if target.name == "status.json":
                original_write(target.parent / "makeup.json",
                               {**metadata, "makeup_finished_at": value["makeup_finished_at"]})
        return original_write(path, value)

    def selected_attempt(path):
        if force_stage is not None and Path(path).resolve() == force_stage.resolve():
            return None
        return original_attempt(path)

    sweep.read_json, sweep.write_json, sweep.attempt_result = safe_read, tagged_write, selected_attempt
    try:
        yield
    finally:
        sweep.read_json, sweep.write_json, sweep.attempt_result = original_read, original_write, original_attempt


def run_makeup(project: Path, list_path: Path, *, dry_run=False, force=False, interval=10.0):
    if interval <= 0:
        raise ValueError("Progress interval must be positive")
    run, frozen, entries, list_sha = load_plan(list_path, project)
    print_plan(run, frozen, entries, list_path, force)
    if dry_run:
        print("Plan only: no workers or inference started.", flush=True)
        return 0
    # Keep the saved protocol and safety limits; don't silently change batch or RAM limits.
    config = dict(frozen, retry_failed=True, progress_interval=interval)
    plan = dict(config, model_order=list(dict.fromkeys(x["model"] for x in entries)),
                modes=list(dict.fromkeys(x["mode"] for x in entries)))
    sweep.preflight(plan)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    progress_path = within_run(run, run / "makeup_progress.json")
    console = ConsoleProgress(interval)
    status = {"kind": "makeup", "makeup_id": run_id, "list_sha256": list_sha,
              "started_at": now(), "planned_routes": len(entries), "counts": {},
              "status": "running", "phase": "GPU", "active": None, "results": [], "selection": entries}

    def publish(**updates):
        status.update(updates, updated_at=now())
        write_json(progress_path, status)
        console(status)

    def execute_stage(stage, job, *, force_run=False):
        metadata = {"is_makeup": True, "execution_kind": "makeup", "makeup_id": run_id,
                    "makeup_list_sha256": list_sha, "makeup_started_at": now(),
                    "makeup_list": str(list_path)}
        with safe_sweep_io(run, stage, metadata, stage if force_run else None):
            return sweep.execute(config, stage, job, publish)

    results = []
    with sweep.exclusive_lock(within_run(run, run / "supervisor.lock")):
        # A killed supervisor may leave a worker alive; do not share its device.
        with sweep.exclusive_lock(within_run(run, run / "worker.lock")):
            pass
        validate_environment(project, run, config, entries)
        try:
            for entry in entries:
                model, batch, mode = entry["model"], entry["batch"], entry["mode"]
                phase = "CPU" if mode in CPU_MODES else "GPU"
                publish(phase=phase, active=None, last_result=None)
                group = within_run(run, run / f"batch_{batch:03d}" / model)
                weights = within_run(run, group / "models" / "checkpoint.pt")
                retained = within_run(run, config["weights"][model]["retained_path"])
                if not weights.is_file():
                    # A selected, never-started group may lack its retained checkpoint link.
                    import os
                    import shutil
                    weights.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(retained, weights)
                    except OSError:
                        shutil.copyfile(retained, weights)
                if digest(weights) != config["weights"][model]["sha256"]:
                    raise ValueError(f"Retained checkpoint hash mismatch: {model}")
                jobs = {}
                for name, job, dependency in sweep.job_plan(config, model, batch, weights, [mode]):
                    stage = within_run(run, group / ("models" if job["kind"] == "export" else "results") / name)
                    reuse = False
                    if dependency and jobs[dependency]["status"] != "passed":
                        stage.mkdir(parents=True, exist_ok=True)
                        result = {"status": "dependency_failed", "dependency": dependency,
                                  "error": "Required ONNX export failed; inference not executed",
                                  "is_makeup": True, "execution_kind": "makeup", "makeup_id": run_id,
                                  "makeup_list_sha256": list_sha, "makeup_started_at": now(),
                                  "makeup_finished_at": now(), "makeup_list": str(list_path)}
                        write_json(stage / "latest.json", result)
                    else:
                        if dependency:
                            job["graph"] = jobs[dependency]["graph"]
                        with safe_sweep_io(run):
                            previous = sweep.attempt_result(stage)
                            reuse = (job["kind"] == "route" and not force and previous
                                     and previous.get("status") == "passed" and previous.get("is_makeup") is True
                                     and previous.get("makeup_list_sha256") == list_sha
                                     and sweep.valid_previous(previous, job))
                        result = previous if reuse else execute_stage(stage, job, force_run=job["kind"] == "route")
                    jobs[name] = result
                    if job["kind"] == "route":
                        results.append({**entry, "status": result["status"], "attempt": result.get("attempt"),
                                        "error": result.get("error"), "reused": bool(reuse)})
                        status["results"] = results
                        status["counts"] = dict(Counter(item["status"] for item in results))
                        publish(active=None, last_result={"job": str(stage), "model": model, "batch": batch,
                                                         "mode": mode, "status": result["status"],
                                                         "error": result.get("error"), "directory": str(stage)})
                # Existing comparison files remain in the original group; no other groups are modified.
                try:
                    with safe_sweep_io(run):
                        sweep.summarize_group(group, config["modes"])
                except (OSError, ValueError, KeyError) as exc:
                    print(f"Comparison not updated for {model} batch {batch}: {exc}", flush=True)
            final = "completed" if all(item["status"] == "passed" for item in results) else "completed_with_failures"
            publish(status=final, active=None, finished_at=now())
            print(f"Makeup {final}: {dict(Counter(x['status'] for x in results))}. Results: {run}", flush=True)
            return 0 if final == "completed" else 1
        except BaseException as exc:
            publish(status="interrupted", active=None, finished_at=now(), error=f"{type(exc).__name__}: {exc}")
            raise
