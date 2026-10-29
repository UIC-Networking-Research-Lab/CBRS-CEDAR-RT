"""Complete missing recorded routes, then rebuild and synchronize the public report."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cbrs_quant.config import CPU_MODES


def now():
    return datetime.now(timezone.utc).isoformat()


def keep_awake(enabled=True):
    """Prevent system sleep only while the experiment is running; do not keep the screen on."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if enabled else 0x80000000)


def wait_for_process(pid):
    import psutil
    keep_awake()
    try:
        process = psutil.Process(pid)
        started = process.create_time()
        while process.is_running() and process.create_time() == started:
            time.sleep(30)
    except psutil.NoSuchProcess:
        pass
    finally:
        keep_awake(False)


def pending(run, batches, include_cpu=False):
    metadata = [json.loads(p.read_text(encoding="utf-8")) for p in run.glob("*/model.json")]
    by_model = {m["model_id"]: m for m in metadata}
    first = metadata[0]["experiment"]
    result = []
    for cpu in (False, True):
        if cpu and not include_cpu:
            continue
        for batch in batches:
            for model in first["model_order"]:
                meta = by_model[model]
                for mode in first["modes"]:
                    if (mode in CPU_MODES) != cpu:
                        continue
                    status = meta.get("routes", {}).get(f"{batch}:{mode}", {}).get("status", "missing")
                    if status != "passed":
                        result.append({"model": model, "batch": batch, "mode": mode, "reason": status})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--batches", nargs="+", type=int, default=[1,4,8])
    parser.add_argument("--include-cpu", action="store_true")
    parser.add_argument("--sync-to", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch-pid", type=int, help="Keep an already-running completion process awake until it exits")
    parser.add_argument("--after-pid", type=int, help="Wait for this completion process, then recompute missing routes")
    args = parser.parse_args(argv)
    if args.watch_pid:
        wait_for_process(args.watch_pid)
        return 0
    if not args.run:
        parser.error("--run is required unless --watch-pid is supplied")
    if args.after_pid:
        print(f"Waiting for completion process {args.after_pid}; missing routes will be recomputed afterwards.",flush=True)
        wait_for_process(args.after_pid)
    run = Path(args.run)
    run = (PROJECT / run).resolve() if not run.is_absolute() else run.resolve()
    selected = pending(run, args.batches, args.include_cpu)
    print(f"Missing routes: {len(selected)}; minimum formal timing: {len(selected)*.15:.1f} hours", flush=True)
    if args.dry_run:
        print(json.dumps(selected, indent=2))
        return 0
    folder = PROJECT / "analysis_outputs" / "batch_completion"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    status_path = folder / "status.json"
    status = {"pid": os.getpid(), "started_at": now(), "run": run.name,
              "batches": args.batches, "include_cpu": args.include_cpu,
              "requested_routes": len(selected), "state": "preparing"}
    def save(**changes):
        status.update(changes, updated_at=now())
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        os.replace(temporary, status_path)
        queue_path = folder / "queue.json"
        if queue_path.exists():
            queue = json.loads(queue_path.read_text(encoding="utf-8-sig"))
            if os.getpid() in queue.get("active_controllers", []):
                queue.update(state=status["state"], updated_at=status["updated_at"])
                if "remaining_routes" in status:
                    queue["remaining_routes"] = status["remaining_routes"]
                if status["state"] in ("complete", "finished_with_missing_routes", "error"):
                    queue["active_controllers"] = []
                queue_tmp = queue_path.with_suffix(".tmp")
                queue_tmp.write_text(json.dumps(queue, indent=2)+"\n", encoding="utf-8")
                os.replace(queue_tmp, queue_path)
    save()
    keep_awake()
    try:
        if selected:
            backup = folder / f"before_{stamp}.zip"
            with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as archive:
                for pattern in ("*/model.json", "*/results.csv", "*/comparison.csv"):
                    for path in run.glob(pattern):
                        archive.write(path, path.relative_to(run).as_posix())
            plan = PROJECT / "makeup" / "batch_completion.json"
            document = {"schema": 1, "created_at": now(), "run": run.relative_to(PROJECT).as_posix(),
                        "description": "All uncompleted selected routes, including pending work; original timing protocol retained.",
                        "experiments": selected}
            plan.write_text(json.dumps(document, indent=2)+"\n", encoding="utf-8")
            save(state="running", plan=str(plan.relative_to(PROJECT)), backup=str(backup.relative_to(PROJECT)))
            result = subprocess.run([sys.executable, "-u", "-B", str(PROJECT/"run_experiments.py"),
                "--makeup", "--makeup-list", str(plan), "--progress-interval", "30"], cwd=PROJECT)
            save(makeup_exit_code=result.returncode)
        remaining = pending(run, args.batches, args.include_cpu)
        save(state="building_report", remaining_routes=remaining)
        report = subprocess.run([sys.executable, "-u", "-B", str(PROJECT/"analyze_results.py"),
            "--run", str(run), "--publish", "results", "--update-readme"], cwd=PROJECT)
        if report.returncode:
            raise RuntimeError(f"Report rebuild exited with {report.returncode}")
        if args.sync_to:
            target = args.sync_to.resolve()
            if target == PROJECT or not all((target/name).is_file() for name in
                    ("requirements.txt", "run_experiments.py", "src/cbrs_quant/rt_report.py")):
                raise ValueError("Sync target must be a separate existing CBRS RT repository")
            shutil.copytree(PROJECT/"results", target/"results", dirs_exist_ok=True)
            files = ["analyze_results.py", "src/cbrs_quant/report_plots.py", "src/cbrs_quant/rt_report.py",
                "src/cbrs_quant/results_store.py", "src/cbrs_quant/report_text.py", "scripts/consolidate_results.py",
                "src/cbrs_quant/analysis_plots.py", "src/cbrs_quant/compact_analysis.py", "tests/test_report.py", "tests/test_completion.py",
                "scripts/complete_batch_sweep.py", "scripts/watch_batch_progress.py", "makeup/batch_completion.json"]
            for name in files:
                if (PROJECT/name).is_file():
                    (target/name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(PROJECT/name,target/name)
            from cbrs_quant.rt_report import update_readme
            update_readme(target,target/"results")
        save(state="complete" if not remaining else "finished_with_missing_routes", finished_at=now())
        print(json.dumps(status,indent=2),flush=True)
        return 0 if not remaining else 1
    except BaseException as exc:
        save(state="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        keep_awake(False)


if __name__ == "__main__":
    raise SystemExit(main())
