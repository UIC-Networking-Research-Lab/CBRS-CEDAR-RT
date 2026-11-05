"""Read committed sweep reports without loading models or touching experiment files."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile

from .config import MODES
from .io import fingerprint, write_json

SCOPES = ("model_compute", "detector_pipeline", "image_to_detection")
ENVIRONMENT_FIELDS = ("gpu", "compute_capability", "l4t", "cuda_build", "python", "platform",
                      "machine", "torch_threads", "torch_interop_threads", "packages",
                      "nvidia_tf32_override")
TIMING_FIELDS = ("protocol_version", "latency_clock", "scope", "confidence", "nms_iou",
                 "max_det", "pool_images", "pool_repeats", "warmup_requests", "minimum_images",
                 "minimum_seconds", "smoke_only_subset")
CLASS_METRICS = ("ap50", "ap50_95", "precision", "recall", "f1", "false_positives_per_image",
                 "matched_frequency_iou")
PROVENANCE_FIELDS = ("execution_kind", "execution_started_at", "makeup_finished_at", "makeup_id")
STATUS_FIELDS = ("model", "batch", "mode", "status", "phase", "attempt", *PROVENANCE_FIELDS, "error")
ACCURACY_FIELDS = ("model", "batch", "mode", *PROVENANCE_FIELDS, "evaluated_images", "map50", "map50_95", "overlap_f1",
                   "map50_delta_pp", "map50_95_delta_pp", "baseline_status",
                   *(f"{name}_{metric}" for name in ("lte", "radar") for metric in CLASS_METRICS))
TIMING_COLUMNS = ("model", "batch", "mode", *PROVENANCE_FIELDS, "scope", "device", "precision", "rounds",
                  "images_per_second_mean", "images_per_second_std", "latency_ms_mean",
                  "p50_ms_mean", "p95_ms_mean", "p99_ms_mean", "speedup_vs_baseline",
                  "baseline_status", "map50_95", "map50_95_delta_pp", "radar_recall",
                  "builder_optimization_level", "provider_nodes")
ROUND_METRICS = ("images_per_second", "latency_ms_mean", "latency_ms_p50", "latency_ms_p95",
                 "latency_ms_p99", "latency_requests", "throughput_images", "throughput_seconds")
ROUND_FIELDS = ("model", "batch", "mode", *PROVENANCE_FIELDS, "scope", "round", *ROUND_METRICS)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def open_snapshot(path, scratch_parent):
    """Windows readers use a temporary hard-link name, leaving the live name replaceable.

    FILE_SHARE_DELETE alone does not make Python's MoveFileEx replacement safe on
    Windows. A same-volume alias pins the old bytes without opening the live name.
    scratch_parent must be outside the experiment; no unsafe read fallback is used.
    """
    if os.name != "nt":
        with path.open("rb") as stream:
            yield stream
        return
    with tempfile.TemporaryDirectory(prefix=".analysis-read-", dir=scratch_parent) as temporary:
        alias = Path(temporary) / "report.json"
        try:
            os.link(path, alias)
        except OSError as exc:
            raise OSError("Windows live analysis requires same-volume hard-link support and "
                          f"a writable run parent: {scratch_parent}") from exc
        with alias.open("rb") as stream:
            yield stream


def read_bytes(path, scratch_parent):
    with open_snapshot(path, scratch_parent) as stream:
        return stream.read()


def select_run(runs_root: Path, explicit: Path | None = None) -> Path:
    """Select by config creation time, not progress-file or directory modification time."""
    if explicit is not None:
        run = explicit.resolve()
        if not (run / "config.json").is_file():
            raise FileNotFoundError(f"No sweep config.json in {run}")
        return run
    candidates = []
    for path in runs_root.glob("*/config.json"):
        try:
            config = json.loads(read_bytes(path, path.parent.parent))
            if not isinstance(config, dict) or not isinstance(config.get("created_at"), str):
                continue
            stamp = datetime.fromisoformat(config["created_at"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            if config.get("schema") == 1 and config.get("model_order"):
                candidates.append((stamp.timestamp(), path.parent.name, path.parent))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not candidates:
        raise FileNotFoundError(f"No readable sweep runs in {runs_root}; use --run <run-directory>")
    return max(candidates)[2].resolve()


def snapshot_directory(run: Path, output_root: Path, runs_root: Path) -> Path:
    """Each invocation owns a new directory outside all experiment output trees."""
    output = (output_root / run.name / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")).resolve()
    if output.is_relative_to(run.resolve()) or output.is_relative_to(runs_root.resolve()):
        raise ValueError("Analysis output must be outside the experiment outputs directory")
    for parent in (output, *output.parents):
        if (parent / "config.json").is_file() and (parent / "source").is_dir():
            raise ValueError(f"Analysis output is inside an experiment: {parent}")
    output.mkdir(parents=True, exist_ok=False)
    return output


class Reader:
    """Hash the exact JSON bytes consumed; model/engine/PNG files are never opened."""

    def __init__(self, root):
        self.root = root.resolve()
        self.files = {}
        self.modified_ns = {}

    def read(self, path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Report path escapes the run: {path}")
        with open_snapshot(path, self.root.parent) as stream:
            raw = stream.read()
            modified_ns = os.fstat(stream.fileno()).st_mtime_ns
        sha = hashlib.sha256(raw).hexdigest()
        if expected is not None and sha != expected:
            raise ValueError(f"Report hash mismatch: {path.name}")
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError(f"Expected a JSON object: {path.name}")
        self.files[path.relative_to(self.root).as_posix()] = sha
        self.modified_ns[path.relative_to(self.root).as_posix()] = modified_ns
        return document


def attempt_name(value):
    # Saved absolute Windows paths must also work after copying results to Linux.
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    if not re.fullmatch(r"attempt_\d+", name):
        raise ValueError(f"Invalid attempt name: {name}")
    return name


def newest_attempt(stage):
    names = [p.name for p in stage.glob("attempt_*")
             if p.is_dir() and re.fullmatch(r"attempt_\d+", p.name)]
    return max(names, key=lambda name: int(name.split("_")[1]), default=None)


def finite(value, name, lower=0, upper=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < lower or (upper is not None and value > upper)):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def identity(report, config, dataset, model, batch, mode):
    meta = report["model"]
    if not isinstance(meta, dict) or not isinstance(report.get("environment"), dict):
        raise ValueError("Report model/environment must be JSON objects")
    expected = (model, batch, mode, config["weights"][model]["sha256"])
    if tuple(meta[k] for k in ("model_id", "batch", "mode", "weights_sha256")) != expected:
        raise ValueError("Report model/batch/mode/checkpoint differs from the sweep")
    if (report["dataset_sha256"] != dataset["records_sha256"]
            or report["profile_sha256"] != dataset["profile_sha256"]
            or report["split"] != config["protocol"]["evaluation_split"]
            or report["smoke_only_subset"] != config["smoke"]):
        raise ValueError("Report dataset/split/profile/smoke protocol differs from the sweep")
    return {key: report["environment"].get(key) for key in ENVIRONMENT_FIELDS}


def validate_reports(reports, config, dataset, model, batch, mode):
    """Return comparison signatures. Reject incomplete, mixed or nonfinite reports."""
    quality = reports["evaluation"]
    env = identity(quality, config, dataset, model, batch, mode)
    metrics = quality["metrics"]
    expected = config.get("protocol", {}).get("quality_protocol")
    if expected and not metrics["protocol"].startswith(expected + ";"):
        raise ValueError("Evaluation metric protocol differs from the experiment configuration")
    images = sum(r["split"] == quality["split"] for r in dataset["records"])
    if config["smoke"]:
        images = min(8, images)
    if quality["kind"] != "evaluation" or quality["evaluated_images"] != images or metrics["images"] != images:
        raise ValueError("Incomplete evaluation set")
    for metric in ("map50", "map50_95"):
        if metrics[metric] is not None:
            finite(metrics[metric], metric, upper=1)
    for name in ("LTE", "Radar"):
        for metric in CLASS_METRICS:
            value = metrics["classes"][name][metric]
            if value is not None:
                finite(value, f"{name}.{metric}", upper=None if metric == "false_positives_per_image" else 1)
    signatures = {"evaluation": fingerprint({
        "environment": env, "protocol": metrics["protocol"], "images": images,
        "collection_confidence": quality["collection_confidence"],
        "operating_confidence": metrics["operating_confidence"],
        "nms_iou": quality["nms_iou"], "max_det": quality["max_det"]})}
    for scope in SCOPES:
        report = reports[scope]
        timing_env = identity(report, config, dataset, model, batch, mode)
        trials = report["rounds"]
        if (report["kind"] != "benchmark" or report["scope"] != scope or report["protocol_version"] != 2
                or report["latency_clock"] != "synchronized_wall"
                or len(trials) != config["protocol"]["rounds"] or timing_env != env):
            raise ValueError(f"Incompatible or incomplete benchmark: {scope}")
        if (report["minimum_seconds"] != config["protocol"]["seconds"]
                or report["warmup_requests"] != (2 if config["smoke"] else 50)
                or report["minimum_images"] != (4 if config["smoke"] else 1000)):
            raise ValueError(f"Timing limits differ from the sweep: {scope}")
        if report["model"].get("route") in ("B", "C"):
            if report["model"].get("builder_optimization_level") != (0 if config["smoke"] else 3):
                raise ValueError("Unexpected TensorRT builder optimization level")
        for trial in trials:
            for field in ROUND_METRICS:
                finite(trial[field], field, lower=1e-12)
            if (trial["latency_requests"] * batch < report["minimum_images"]
                    or trial["throughput_images"] < report["minimum_images"]
                    or trial["throughput_images"] % batch
                    or trial["throughput_seconds"] < report["minimum_seconds"]
                    or not math.isclose(trial["images_per_second"],
                                        trial["throughput_images"] / trial["throughput_seconds"], rel_tol=1e-8)):
                raise ValueError(f"Invalid benchmark counters: {scope}")
        signatures[scope] = fingerprint({**{k: report[k] for k in TIMING_FIELDS}, "environment": env})
    return signatures


def collect(run: Path, progress=print) -> dict:
    reader = Reader(run)
    started = utc_now()
    config = reader.read(run / "config.json")
    if fingerprint({k: v for k, v in config.items() if k != "config_sha256"}) != config["config_sha256"]:
        raise ValueError("Sweep configuration hash mismatch")
    for key in ("model_order", "modes", "batches"):
        values = config[key]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Invalid sweep {key}")
    if (any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in config["model_order"])
            or set(config["modes"]) - set(MODES)
            or any(type(b) is not int or b < 1 for b in config["batches"])
            or config["protocol"]["scopes"] != list(SCOPES)):
        raise ValueError("Unsupported sweep matrix")
    dataset = reader.read(run / "dataset.json", config["manifest_sha256"])
    if (fingerprint(dataset["records"]) != dataset["records_sha256"]
            or fingerprint(dataset["profile"]) != dataset["profile_sha256"]):
        raise ValueError("Dataset manifest content hash mismatch")
    routes, accepted, warnings = [], [], []
    try:
        live = reader.read(run / "progress.json")
    except (OSError, ValueError) as exc:
        live = {}
        warnings.append(f"Progress snapshot unavailable: {exc}")
    for batch in config["batches"]:
        for model in config["model_order"]:
            for mode in config["modes"]:
                stage = run / f"batch_{batch:03d}" / model / "results" / mode
                row = dict(model=model, batch=batch, mode=mode, status="pending", phase="", attempt="", error="")
                routes.append(row)
                try:
                    latest_path = stage / "latest.json"
                    latest = reader.read(latest_path) if latest_path.exists() else {}
                    if latest and (not isinstance(latest.get("status"), str) or latest["status"] not in
                                   ("passed", "failed", "dependency_failed", "interrupted")):
                        raise ValueError("Invalid committed route status")
                    if latest:
                        row.update(execution_kind="makeup" if latest.get("is_makeup") is True else "original",
                                   execution_started_at=latest.get("makeup_started_at") or latest.get("started_at", ""),
                                   makeup_finished_at=latest.get("makeup_finished_at", ""),
                                   makeup_id=latest.get("makeup_id", ""))
                    newest = newest_attempt(stage)
                    committed = attempt_name(latest["attempt"]) if latest.get("attempt") else None
                    if latest.get("status") == "dependency_failed":
                        job_path = stage / newest / "job.json" if newest else None
                        attempt_stamp = (job_path if job_path.is_file() else stage / newest).stat().st_mtime_ns if newest else 0
                        failure_stamp = reader.modified_ns[latest_path.relative_to(run).as_posix()]
                        if attempt_stamp <= failure_stamp:
                            row.update(status="dependency_failed", error=latest.get("error", "Required export failed"))
                            continue
                    if newest and (not committed or int(newest.split("_")[1]) > int(committed.split("_")[1])):
                        row.update(status="incomplete", attempt=(stage / newest).relative_to(run).as_posix())
                        status_path = stage / newest / "status.json"
                        if status_path.exists():
                            state = reader.read(status_path)
                            row.update(phase=state.get("phase", ""), error=state.get("error", ""))
                        makeup_path = stage / newest / "makeup.json"
                        if makeup_path.exists():
                            metadata = reader.read(makeup_path)
                            row.update(execution_kind="makeup", execution_started_at=metadata.get("makeup_started_at", ""),
                                       makeup_finished_at=metadata.get("makeup_finished_at", ""),
                                       makeup_id=metadata.get("makeup_id", ""))
                        continue
                    if not latest:
                        continue
                    row.update(status=latest["status"], phase=latest.get("phase", ""), error=latest.get("error", ""))
                    if committed:
                        row["attempt"] = (stage / committed).relative_to(run).as_posix()
                    if latest["status"] != "passed":
                        continue
                    if not committed or committed != newest:
                        raise ValueError("Committed attempt directory is missing")
                    if not isinstance(latest.get("artifact_sha256"), dict):
                        raise ValueError("Committed artifact hashes must be a JSON object")
                    artifacts = {k.replace("\\", "/"): v for k, v in latest["artifact_sha256"].items()}
                    if any(not isinstance(artifacts.get(f"{name}.json"), str)
                           or not re.fullmatch(r"[0-9a-f]{64}", artifacts[f"{name}.json"])
                           for name in ("evaluation", *SCOPES)):
                        raise ValueError("Missing or invalid committed report hash")
                    reports = {name: reader.read(stage / committed / f"{name}.json", artifacts[f"{name}.json"])
                               for name in ("evaluation", *SCOPES)}
                    signatures = validate_reports(reports, config, dataset, model, batch, mode)
                    original_sha = reader.files[latest_path.relative_to(run).as_posix()]
                    if (hashlib.sha256(read_bytes(latest_path, run.parent)).hexdigest() != original_sha
                            or newest_attempt(stage) != newest):
                        row.update(status="changed_during_read", error="Attempt changed while reading; rerun analysis")
                        continue
                    accepted.append({**row, "reports": reports, "signatures": signatures})
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    row.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
        progress(f"  Read batch={batch}: {len(accepted)} valid routes so far", flush=True)
    return {"schema": 1, "run": str(run), "started_at": started, "finished_at": utc_now(),
            "config": config, "progress_at_start": live, "routes": routes, "accepted": accepted,
            "warnings": warnings, "input_sha256": reader.files}


def tables(snapshot, baseline):
    quality_rows, timing_rows, trial_rows = [], [], []
    references = {(r["model"], r["batch"]): r for r in snapshot["accepted"] if r["mode"] == baseline}
    for item in snapshot["accepted"]:
        key = {k: item.get(k, "") for k in ("model", "batch", "mode", *PROVENANCE_FIELDS)}
        reference = references.get((item["model"], item["batch"]))
        metrics = item["reports"]["evaluation"]["metrics"]
        quality = {**key, "evaluated_images": metrics["images"], "map50": metrics["map50"],
                   "map50_95": metrics["map50_95"], "baseline_status": "missing",
                   "quality_signature": item["signatures"]["evaluation"]}
        quality["overlap_f1"] = metrics.get("overlap", {}).get("f1")
        quality.update({f"{name.lower()}_{metric}": metrics["classes"][name][metric]
                        for name in ("LTE", "Radar") for metric in CLASS_METRICS})
        if reference:
            matched = item["signatures"]["evaluation"] == reference["signatures"]["evaluation"]
            quality["baseline_status"] = "matched" if matched else "protocol_mismatch"
            for name in ("map50", "map50_95"):
                base = reference["reports"]["evaluation"]["metrics"][name]
                if matched and base is not None and metrics[name] is not None:
                    quality[f"{name}_delta_pp"] = 100 * (metrics[name] - base)
        quality_rows.append(quality)
        for scope in SCOPES:
            report = item["reports"][scope]
            trials = report["rounds"]
            rates = [t["images_per_second"] for t in trials]
            row = {**key, "scope": scope, "device": report["model"].get("device"),
                   "precision": report["model"].get("precision"), "rounds": len(trials),
                   "images_per_second_mean": statistics.mean(rates),
                   "images_per_second_std": statistics.pstdev(rates),
                   "latency_ms_mean": statistics.mean(t["latency_ms_mean"] for t in trials),
                   **{f"p{q}_ms_mean": statistics.mean(t[f"latency_ms_p{q}"] for t in trials) for q in (50, 95, 99)},
                   "baseline_status": "missing", "map50_95": quality["map50_95"],
                   "map50_95_delta_pp": quality.get("map50_95_delta_pp"), "radar_recall": quality["radar_recall"],
                   "builder_optimization_level": report["model"].get("builder_optimization_level"),
                   "provider_nodes": json.dumps(report["model"].get("provider_audit", {}).get("node_events", {}), sort_keys=True),
                   "timing_signature": item["signatures"][scope],
                   "quality_signature": item["signatures"]["evaluation"]}
            if reference:
                matched = item["signatures"][scope] == reference["signatures"][scope]
                row["baseline_status"] = "matched" if matched else "protocol_mismatch"
                if matched:
                    base = statistics.mean(t["images_per_second"] for t in reference["reports"][scope]["rounds"])
                    row["speedup_vs_baseline"] = row["images_per_second_mean"] / base
            timing_rows.append(row)
            trial_rows.extend({**key, "scope": scope, "round": i + 1, **trial} for i, trial in enumerate(trials))
    return quality_rows, timing_rows, trial_rows


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(snapshot, output, baseline, quality, timings, trials, figures):
    """Write a browsable local report; missing values stay blank, never become zero."""
    counts = dict(Counter(r["status"] for r in snapshot["routes"]))
    planned = len(snapshot["routes"])
    accepted = len(snapshot["accepted"])
    failures = [r for r in snapshot["routes"] if r["status"] in ("failed", "dependency_failed", "interrupted", "invalid")]
    for name, rows, fields in (("routes", snapshot["routes"], STATUS_FIELDS),
                              ("failures", failures, STATUS_FIELDS),
                              ("accuracy", quality, ACCURACY_FIELDS),
                              ("performance", timings, TIMING_COLUMNS),
                              ("rounds", trials, ROUND_FIELDS)):
        write_csv(output / f"{name}.csv", rows, fields)
    summary = {"source_run": snapshot["run"], "snapshot_started_at": snapshot["started_at"],
               "snapshot_finished_at": snapshot["finished_at"], "planned_routes": planned,
               "valid_routes": accepted, "counts": counts, "baseline": baseline,
               "smoke": snapshot["config"]["smoke"], "warnings": snapshot["warnings"],
               "performance_rows": len(timings), "figures": figures,
               "speedups_missing_baseline": sum(r["baseline_status"] == "missing" for r in timings),
               "speedups_protocol_mismatch": sum(r["baseline_status"] == "protocol_mismatch" for r in timings)}
    write_json(output / "summary.json", summary)
    write_json(output / "snapshot.json", snapshot)
    escape = lambda value: html.escape(str(value))
    gallery = []
    for figure in figures:
        links = " · ".join(f'<a href="{escape(p)}">{Path(p).suffix[1:].upper()}</a>' for p in figure["files"])
        png = next((p for p in figure["files"] if p.endswith(".png")), None)
        preview = f'<img loading="lazy" src="{escape(png)}" alt="{escape(figure["title"])}">' if png else ""
        gallery.append(f'<section><h2>{escape(figure["title"])}</h2>{links}{preview}</section>')
    failure_table = "".join("<tr>" + "".join(f"<td>{escape(r.get(k, ''))}</td>" for k in STATUS_FIELDS) + "</tr>" for r in failures)
    notes = "".join(f"<li>{escape(w)}</li>" for w in snapshot["warnings"])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>CBRS analysis — {escape(Path(snapshot['run']).name)}</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:36px auto;padding:0 24px;color:#243444}}
h1,h2{{line-height:1.2}}h2{{font-size:20px}}section{{margin:32px 0;border-top:1px solid #ddd;padding-top:12px}}
img{{display:block;width:100%;height:auto;margin-top:12px}}code{{overflow-wrap:anywhere}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border:1px solid #ddd;text-align:left;overflow-wrap:anywhere}}
a{{color:#0072b2}}.status{{background:#edf4f8;padding:16px;border-radius:8px}}</style>
<h1>CBRS experiment analysis</h1><p><code>{escape(snapshot['run'])}</code></p>
<p class="status"><b>{accepted}/{planned} valid routes</b> · {escape(counts)}<br>
Read interval (UTC): {escape(snapshot['started_at'])} — {escape(snapshot['finished_at'])}<br>
{'SMOKE DATA — not formal measurements' if snapshot['config']['smoke'] else 'Formal protocol; partial results until all routes finish'}</p>
<p>Downloads: {' · '.join(f'<a href="{name}.csv">{name}.csv</a>' for name in ('routes','failures','performance','accuracy','rounds'))}
 · <a href="summary.json">summary.json</a> · <a href="snapshot.json">snapshot.json</a></p>
<ul><li>Only committed successful routes with valid report hashes are analyzed. Newer unfinished retries supersede older results.
"Incomplete" means uncommitted on disk; it does not prove that a worker is still alive.</li>
<li>Baseline: <code>{escape(baseline)}</code>, matched model, batch, scope, environment and protocol. Missing or mismatched baselines leave ratios/deltas blank.</li>
<li>Throughput is mean ± population standard deviation across rounds, not a confidence interval.
Latency is per batch request; P95 is the mean of round P95 values, not a pooled percentile.</li>
<li>Accuracy deltas are current minus baseline, in percentage points. Negative means degradation.
AP uses the project's 101-point protocol, not the complete COCO evaluator.</li>
<li>model_compute uses resident tensors; detector_pipeline includes tensor preparation, transfers and detection postprocessing;
image_to_detection also includes PNG reading/decoding with a warm file cache.</li>
<li>Missing measurements are gaps. No cross-model average speedup or total GPU-memory ranking is inferred.</li>
<li>This is a read-interval snapshot, not a lock on the live experiment. JSON hashes are checked; checkpoints, engines and PNGs are not rehashed.</li>{notes}</ul>
<h2>Failures / invalid reports ({len(failures)})</h2>
<table><tr>{''.join(f'<th>{k}</th>' for k in STATUS_FIELDS)}</tr>{failure_table}</table>
{''.join(gallery)}</html>'''
    (output / "index.html").write_text(document, encoding="utf-8")
    return summary
