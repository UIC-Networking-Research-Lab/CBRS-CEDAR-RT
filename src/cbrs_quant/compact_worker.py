"""Run one existing export or inference worker and return its reports in memory."""

from __future__ import annotations

import random
import shutil
import traceback
from pathlib import Path


def run(task, connection):
    from . import runner, sweep
    from .compact import save_graph
    from .io import read_json, write_json

    scratch = Path(task["scratch"])
    attempt = scratch / "attempt"
    attempt.mkdir()
    job = dict(task["job"], output=str(scratch))
    job_path = attempt / "job.json"
    write_json(job_path, job)
    original_state, original_predict = sweep.stage_state, runner.predict

    def progress(path, **updates):
        original_state(path, **updates)
        if "phase" in updates or "detail" in updates:
            connection.send({"type": "progress", "phase": updates.get("phase", ""),
                             "detail": updates.get("detail", {})})

    def demo(backend, paths, output, *args, **kwargs):
        if not task.get("capture_demo"):
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "predictions.json", {"records": []})
            return {"records": []}
        manifest = read_json(job["manifest"])
        choices = [Path(job["data"]) / r["image"] for r in manifest["records"] if r["split"] == "val"]
        selected = random.Random(42).sample(choices, min(10, len(choices)))
        return original_predict(backend, selected, output, *args, **kwargs)

    sweep.stage_state, runner.predict = progress, demo
    try:
        try:
            sweep._run_worker(job_path)
        except Exception:
            error = traceback.format_exc()
        else:
            error = ""
        status_path = attempt / "status.json"
        state = read_json(status_path) if status_path.exists() else {"status": "failed", "error": error}
        if error:
            state.update(status="failed", traceback=error)
        reports = {}
        if job["kind"] == "export" and state.get("status") == "passed":
            target = save_graph(Path(task["model_dir"]), job["batch"], job["precision"], state["graph"])
            state["graph"] = str(target)
        elif job["kind"] == "route":
            for name in ("evaluation", "smoke", "model_compute", "detector_pipeline", "image_to_detection"):
                file = attempt / f"{name}.json"
                if file.is_file():
                    reports[name] = read_json(file)
            if task.get("capture_demo") and (attempt / "demo/predictions.json").is_file():
                predictions = read_json(attempt / "demo/predictions.json")
                target = Path(task["model_dir"]) / "demo"
                target.mkdir(exist_ok=True)
                for old in target.glob("*.png"):
                    old.unlink()
                images = sorted((attempt / "demo").glob("*.png"))
                for image in images:
                    shutil.copyfile(image, target / image.name)
                state["demo"] = {"seed": 42, "saved_images": len(images), "source_mode": job["mode"],
                                 "batch": job["batch"], "images": [p.name for p in images],
                                 "predictions": predictions.get("records", [])}
        connection.send({"type": "result", "state": state, "reports": reports})
    finally:
        sweep.stage_state, runner.predict = original_state, original_predict
        connection.close()
