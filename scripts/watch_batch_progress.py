"""Publish lightweight live progress without rerunning analysis or touching workers."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

PROJECT = Path(__file__).resolve().parents[1]


def snapshot(run, batches, include_cpu=False):
    rows = []
    for path in sorted(run.glob("*/model.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        for batch in batches:
            for mode in meta["experiment"]["modes"]:
                if not include_cpu and "cpu" in mode:
                    continue
                route = meta.get("routes", {}).get(f"{batch}:{mode}", {})
                rows.append({"model": meta["model_id"], "batch": batch, "policy": mode,
                             "device": "CPU" if "cpu" in mode else "GPU",
                             "status": route.get("status", "missing"),
                             "error": re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+", "<USER_HOME>", str(route.get("error", "")))})
    if not rows:
        raise ValueError(f"No model metadata in {run}")
    return rows


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".progress.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def publish(target, rows, batches, alive, include_cpu=False, stopped_by_user=False):
    output = target / "results"
    output.mkdir(parents=True, exist_ok=True)
    completed = sum(r["status"] == "passed" for r in rows)
    failed = sum(r["status"] == "failed" for r in rows)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = "complete" if completed == len(rows) else "running" if alive else "stopped_with_missing_routes"
    if stopped_by_user:
        state = "stopped_by_user"
    data = {"updated_at_utc": timestamp, "state": state, "batches": batches,
            "include_cpu": include_cpu, "passed": completed, "planned": len(rows),
            "failed": failed, "remaining": len(rows) - completed, "active_controllers": alive}
    devices = ("GPU", "CPU") if include_cpu else ("GPU",)
    scope = "GPU and CPU policies" if include_cpu else "GPU policies only; CPU testing is excluded"
    lines = ["# Batch completion progress", "", f"Updated: **{timestamp}** (UTC).", "",
             f"Status: **{state}**. Completed **{completed}/{len(rows)}** requested routes; "
             f"**{len(rows)-completed}** remain, including **{failed}** recorded failures.", "",
             f"Scope: batches **{', '.join(map(str,batches))}**, {scope}. "
             "Batch 16 and 32 entries in the full experiment plan are outside this makeup run.", "",
             ("Selected experiments are running." if alive else "Experiments are stopped. Completed measurements are retained."), "",
             "| Batch | " + " | ".join(f"{d} completed / planned" for d in devices) + " | Total remaining |",
             "|---|" + "---:|" * (len(devices) + 1)]
    for batch in batches:
        selected = [r for r in rows if r["batch"] == batch]
        cells = []
        for device in devices:
            group = [r for r in selected if r["device"] == device]
            cells.append(f"{sum(r['status']=='passed' for r in group)} / {len(group)}")
        lines.append(f"| {batch} | {' | '.join(cells)} | {sum(r['status']!='passed' for r in selected)} |")
    recorded = output / "route_status.csv"
    if recorded.exists():
        with recorded.open(encoding="utf-8-sig", newline="") as stream:
            published = [r for r in csv.DictReader(stream) if int(r["batch"]) in batches
                         and (include_cpu or "cpu" not in r["mode"])]
        report_count = sum(r["status"] == "passed" for r in published)
        lines.extend(["", "## Report snapshot", "",
            f"The published figures and tables currently contain **{report_count}/{len(published)}** completed "
            "routes within this scope. Live progress above is read directly from experiment metadata.", "",
            ("The controllers rebuild and synchronize the complete report after each queue finishes. "
             "This page refreshes every minute while the controllers are active; it does not change measurements."
             if alive else "Automatic updates are stopped. The published report remains the last generated snapshot; "
             "saved completed measurements have not been removed.")])
    errors = [r for r in rows if r["status"] == "failed"]
    if errors:
        lines.extend(["", "## Recorded failures", "", "| Model | Batch | Policy | Error |", "|---|---:|---|---|"])
        for r in errors:
            error = r["error"].replace("|", "/").replace("\n", " ")
            lines.append(f"| {r['model']} | {r['batch']} | {r['policy']} | {error} |")
    atomic_text(output / "PROGRESS.md", "\n".join(lines) + "\n")
    atomic_text(output / "progress.json", json.dumps(data, indent=2) + "\n")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--target", type=Path, action="append", required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1,4,8])
    parser.add_argument("--include-cpu", action="store_true")
    parser.add_argument("--stopped-by-user", action="store_true", help="Mark a one-off snapshot as explicitly stopped")
    parser.add_argument("--pids", nargs="+", type=int, default=[])
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.stopped_by_user and (not args.once or args.pids):
        parser.error("--stopped-by-user requires --once without --pids")
    import psutil
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    tracked = {}
    for pid in args.pids:
        try:
            tracked[pid] = psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:
            pass
    while True:
        alive = []
        for pid, started in tracked.items():
            try:
                if psutil.Process(pid).create_time() == started:
                    alive.append(pid)
            except psutil.NoSuchProcess:
                pass
        try:
            rows = snapshot(args.run, args.batches, args.include_cpu)
            for target in args.target:
                data = publish(target, rows, args.batches, alive, args.include_cpu, args.stopped_by_user)
            print(json.dumps(data), flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"Snapshot deferred: {type(exc).__name__}: {exc}", flush=True)
        if args.once or not alive:
            break
        time.sleep(max(10,args.interval))


if __name__ == "__main__":
    main()
