"""The compact per-model result format shared by experiments and migration."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from .analysis import ENVIRONMENT_FIELDS, SCOPES, TIMING_FIELDS
from .io import digest, fingerprint, read_json, sidecar, write_json

CLASS_FIELDS = ("targets", "ap50", "ap50_95", "precision", "recall", "f1",
                "true_positives", "false_positives", "false_positives_per_image", "matched_frequency_iou")
ROUND_FIELDS = ("latency_requests", "latency_ms_mean", "latency_ms_p50", "latency_ms_p95",
                "latency_ms_p99", "throughput_images", "throughput_seconds", "images_per_second")
RESULT_FIELDS = ("model", "batch", "mode", "status", "scope", "round", "execution_kind",
                 "started_at", "finished_at", "elapsed_seconds", "phase", "error", "smoke_passed",
                 "smoke_records", "evaluated_images", "map50", "map50_95",
                 *(f"{name}_{field}" for name in ("lte", "radar") for field in CLASS_FIELDS),
                 "load_seconds", "builder_optimization_level", "timing_signature", "quality_signature", *ROUND_FIELDS)
COMPARISON_FIELDS = ("model", "batch", "mode", "scope", "rounds", "images_per_second_mean",
                     "images_per_second_std", "p50_ms_mean", "p95_ms_mean", "p99_ms_mean",
                     "speedup_vs_torch_fp32", "map50", "map50_95", "map50_95_delta_pp",
                     "radar_recall", "builder_optimization_level", "timing_signature", "quality_signature", "baseline_status")
CORE_FILES = tuple(f"cbrs_quant/{name}.py" for name in
                   ("config", "data", "engine", "export", "io", "metrics", "models", "ort_backend",
                    "postprocess", "runner", "runtime", "smoke", "sweep")) + ("cbrs_quant/vendor/asym_yolo.py",)
EXPORT_FILES = tuple(name for name in CORE_FILES if name.endswith(
    ("config.py", "data.py", "export.py", "io.py", "models.py", "sweep.py", "asym_yolo.py")))


def code_signature(hashes, names=CORE_FILES):
    return fingerprint({name: hashes.get(name) for name in names})


def current_signature(names=CORE_FILES):
    root = Path(__file__).resolve().parents[1]
    hashes = {name: digest(root / name) for name in names if (root / name).is_file()}
    # The CLI's default dataset location does not alter the worker or export.
    # Normalize this path-only change so existing experiment signatures remain comparable.
    worker = root / "cbrs_quant/sweep.py"
    if "cbrs_quant/sweep.py" in hashes:
        content = worker.read_bytes().replace(b'default=Path("data/yolo_split")',
                                              b'default=Path("../yolo_split")')
        hashes["cbrs_quant/sweep.py"] = hashlib.sha256(content).hexdigest()
    return code_signature(hashes, names)


def route_key(batch, mode):
    return f"{batch}:{mode}"


def graph_name(batch, precision):
    return f"batch_{batch:03d}_{precision}.onnx"


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    # A temporary file, not a persistent output folder, protects partial reads.
    import os
    import tempfile
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def timing_signature(report, code=None):
    model = report.get("model", {})
    environment = report.get("environment", {})
    return fingerprint({"protocol": {key: report.get(key) for key in TIMING_FIELDS},
                        "environment": {key: environment.get(key) for key in ENVIRONMENT_FIELDS},
                        "dataset": report.get("dataset_sha256"), "profile": report.get("profile_sha256"),
                        "split": report.get("split"), "weights": model.get("weights_sha256"),
                        "rounds": len(report.get("rounds", [])), "code": code})


def quality_signature(report, code=None):
    model = report.get("model", {})
    environment = report.get("environment", {})
    metrics = report.get("metrics", {})
    return fingerprint({"environment": {key: environment.get(key) for key in ENVIRONMENT_FIELDS},
                        "protocol": metrics.get("protocol"), "images": metrics.get("images"),
                        "collection_confidence": report.get("collection_confidence"),
                        "operating_confidence": metrics.get("operating_confidence"),
                        "nms_iou": report.get("nms_iou"), "max_det": report.get("max_det"),
                        "dataset": report.get("dataset_sha256"), "profile": report.get("profile_sha256"),
                        "split": report.get("split"), "weights": model.get("weights_sha256"),
                        "code": code})


def slim(report, excluded):
    return {key: value for key, value in report.items() if key not in excluded}


def route_rows(model, batch, mode, status, reports, info):
    evaluation = reports.get("evaluation", {})
    metrics = evaluation.get("metrics", {})
    smoke = reports.get("smoke", {})
    base = {"model": model, "batch": batch, "mode": mode, "status": status,
            "execution_kind": info.get("execution_kind", "original"),
            "started_at": info.get("started_at", ""), "finished_at": info.get("finished_at", ""),
            "elapsed_seconds": info.get("elapsed_seconds", ""), "phase": info.get("phase", ""),
            "error": info.get("error", ""), "smoke_passed": smoke.get("passed", ""),
            "smoke_records": json_text(smoke.get("records", [])) if smoke else "",
            "evaluated_images": evaluation.get("evaluated_images", metrics.get("images", "")),
            "map50": metrics.get("map50", ""), "map50_95": metrics.get("map50_95", ""),
            "load_seconds": evaluation.get("load_seconds", ""),
            "quality_signature": quality_signature(evaluation, info.get("inference_code_sha256")) if evaluation else ""}
    for name in ("LTE", "Radar"):
        values = metrics.get("classes", {}).get(name, {})
        base.update({f"{name.lower()}_{field}": values.get(field, "") for field in CLASS_FIELDS})
    rows = []
    for scope in SCOPES:
        report = reports.get(scope, {})
        for number, record in enumerate(report.get("rounds", []), 1):
            rows.append({**base, "scope": scope, "round": number,
                         "builder_optimization_level": report.get("model", {}).get("builder_optimization_level", ""),
                         "timing_signature": timing_signature(report, info.get("inference_code_sha256")), **record})
    return rows or [base]


def put_route(model_dir, batch, mode, status, reports, info):
    meta_path, results_path = model_dir / "model.json", model_dir / "results.csv"
    meta = read_json(meta_path)
    rows = [row for row in read_csv(results_path) if not (int(row["batch"]) == batch and row["mode"] == mode)]
    rows.extend(route_rows(meta["model_id"], batch, mode, status, reports, info))
    route = {"status": status, **info, "timing_signatures": {}, "quality_signature": None}
    for name, report in reports.items():
        if name == "smoke":
            route["smoke"] = slim(report, {"model", "environment", "records"})
        elif name == "evaluation":
            route["evaluation"] = slim(report, {"model", "environment", "metrics"})
            route["evaluation"]["metrics_protocol"] = report.get("metrics", {}).get("protocol")
            route["evaluation"]["operating_confidence"] = report.get("metrics", {}).get("operating_confidence")
            route["quality_signature"] = quality_signature(report, info.get("inference_code_sha256"))
        elif name in SCOPES:
            route.setdefault("benchmarks", {})[name] = slim(report, {"model", "environment", "rounds"})
            route["timing_signatures"][name] = timing_signature(report, info.get("inference_code_sha256"))
        model = report.get("model", {})
        if model:
            details = dict(model)
            details.pop("environment", None)
            calibration = details.pop("calibration", None)
            if calibration:
                key = fingerprint(calibration)
                meta.setdefault("calibrations", {})[key] = calibration
                details["calibration_ref"] = key
            layers = details.pop("layers", None)
            if layers:
                key = fingerprint(layers)
                meta.setdefault("engine_inspections", {})[key] = layers
                details["engine_inspection_ref"] = key
            route["backend"] = details
        if report.get("environment"):
            env = report["environment"]
            key = fingerprint(env)
            meta.setdefault("environments", {})[key] = env
            route["environment"] = key
    meta.setdefault("routes", {})[route_key(batch, mode)] = route
    write_csv(results_path, rows, RESULT_FIELDS)
    write_json(meta_path, meta)
    write_csv(model_dir / "comparison.csv", comparisons(meta, rows), COMPARISON_FIELDS)


def comparisons(meta, rows, baseline="torch_cuda_fp32"):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") == "passed" and row.get("scope") and row.get("images_per_second") not in (None, ""):
            grouped[(int(row["batch"]), row["mode"], row["scope"])].append(row)
    result = []
    for (batch, mode, scope), trials in grouped.items():
        rate = [float(row["images_per_second"]) for row in trials]
        first = trials[0]
        item = {"model": meta["model_id"], "batch": batch, "mode": mode, "scope": scope,
                "rounds": len(trials), "images_per_second_mean": statistics.mean(rate),
                "images_per_second_std": statistics.pstdev(rate),
                **{f"p{n}_ms_mean": statistics.mean(float(row[f"latency_ms_p{n}"]) for row in trials) for n in (50, 95, 99)},
                "map50": first.get("map50", ""), "map50_95": first.get("map50_95", ""),
                "radar_recall": first.get("radar_recall", ""),
                "builder_optimization_level": first.get("builder_optimization_level", ""),
                "timing_signature": first.get("timing_signature", ""),
                "quality_signature": first.get("quality_signature", ""), "baseline_status": "missing"}
        reference = grouped.get((batch, baseline, scope))
        this = meta.get("routes", {}).get(route_key(batch, mode), {})
        base = meta.get("routes", {}).get(route_key(batch, baseline), {})
        if reference:
            match = (first.get("timing_signature") and first.get("timing_signature")
                     == reference[0].get("timing_signature"))
            item["baseline_status"] = "matched" if match else "protocol_mismatch"
            if match:
                item["speedup_vs_torch_fp32"] = item["images_per_second_mean"] / statistics.mean(
                    float(row["images_per_second"]) for row in reference)
            if (first.get("quality_signature") and first.get("quality_signature") == reference[0].get("quality_signature")
                    and first.get("map50_95") not in (None, "") and reference[0].get("map50_95") not in (None, "")):
                item["map50_95_delta_pp"] = (float(first["map50_95"]) - float(reference[0]["map50_95"])) * 100
        result.append(item)
    return result


def initialize_model(output, model_id, config, dataset, checkpoint_sha, source_run=None):
    directory = output / model_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.json"
    if not path.exists():
        meta = {"schema": 1, "model_id": model_id, "experiment": config,
                "checkpoint_sha256": checkpoint_sha, "source_run": source_run,
                "dataset": {"profile": dataset.get("profile"), "records_sha256": dataset.get("records_sha256"),
                            "profile_sha256": dataset.get("profile_sha256"),
                            "split_counts": dict(Counter(row["split"] for row in dataset.get("records", [])))},
                "onnx": {}, "routes": {}, "environments": {}, "demo": {}}
        write_json(path, meta)
        write_csv(directory / "results.csv", [], RESULT_FIELDS)
        write_csv(directory / "comparison.csv", [], COMPARISON_FIELDS)
        (directory / "demo").mkdir(exist_ok=True)
    return directory


def copy_checkpoint(model_dir, checkpoint):
    """Keep the original PT checkpoint beside its ONNX exports."""
    meta_path = model_dir / "model.json"
    meta = read_json(meta_path)
    checkpoint = Path(checkpoint).resolve()
    target = model_dir / "best.pt"
    if checkpoint != target.resolve():
        shutil.copyfile(checkpoint, target)
    if digest(target) != meta["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
    meta["checkpoint_source"] = str(checkpoint)
    meta["checkpoint_path"] = target.name
    write_json(meta_path, meta)


def save_graph(model_dir, batch, precision, graph, export_code_sha256=None):
    graph = Path(graph)
    graph_meta = read_json(sidecar(graph))
    if graph_meta.get("sha256") != digest(graph):
        raise ValueError(f"ONNX hash mismatch: {graph}")
    target = model_dir / graph_name(batch, precision)
    shutil.copyfile(graph, target)
    shutil.copyfile(sidecar(graph), sidecar(target))
    meta = read_json(model_dir / "model.json")
    meta["onnx"][f"{batch}:{precision}"] = {"file": target.name, "metadata": sidecar(target).name,
                                             "sha256": graph_meta.get("sha256"),
                                             "precision": graph_meta.get("graph_precision"),
                                             "export_code_sha256": export_code_sha256 or current_signature(EXPORT_FILES)}
    write_json(model_dir / "model.json", meta)
    return target


