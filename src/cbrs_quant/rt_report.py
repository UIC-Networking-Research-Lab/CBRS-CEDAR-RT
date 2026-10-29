"""Rebuild a portable, complete RT report from recorded compact experiments only."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import re

from .analysis import SCOPES, read_bytes
from .compact import comparisons
from .report_plots import coverage_chart, model_chart, model_label, overview


def sha(data):
    return hashlib.sha256(data).hexdigest()


def csv_rows(data):
    return list(csv.DictReader(StringIO(data.decode("utf-8-sig"))))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def sanitize(value):
    """Remove private home prefixes from public provenance, leaving measurements intact."""
    if isinstance(value, dict):
        return {k: sanitize(v) for k,v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", "<USER_HOME>", value)
        return re.sub(r"/(?:home|Users)/[^/\s\"']+", "<USER_HOME>", value)
    return value


def number(value):
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite recorded metric: {value}")
    return number


def enrich(meta, raw, baseline="torch_cuda_fp32"):
    """Validate each route before aggregation; never mix rounds from different protocols."""
    groups = defaultdict(list)
    for row in raw:
        if row.get("status") == "passed" and row.get("scope"):
            groups[(int(row["batch"]), row["mode"], row["scope"])].append(row)
    expected = int(meta.get("experiment", {}).get("protocol", {}).get("rounds", 3))
    for key, trials in groups.items():
        numbers = [int(r["round"]) for r in trials]
        if sorted(numbers) != list(range(1, expected+1)):
            raise ValueError(f"{meta['model_id']} {key}: expected {expected} unique timing rounds")
        if len({(r.get("timing_signature"), r.get("quality_signature")) for r in trials}) != 1:
            raise ValueError(f"{meta['model_id']} {key}: mixed timing/quality signatures")
        for field in ("evaluated_images", "map50", "map50_95", "lte_f1", "radar_f1", "overlap_f1"):
            if len({r.get(field, "") for r in trials}) != 1:
                raise ValueError(f"{meta['model_id']} {key}: inconsistent {field} across rounds")
        if any(number(r["images_per_second"]) <= 0 for r in trials):
            raise ValueError(f"{meta['model_id']} {key}: non-positive throughput")
    result = comparisons(meta, raw, baseline)
    for row in result:
        trials = groups[(int(row["batch"]), row["mode"], row["scope"])]
        first = trials[0]
        rate = float(row["images_per_second_mean"])
        row["inference_time_per_frame_s"] = 1.0/rate
        row["lte_f1"], row["radar_f1"] = number(first.get("lte_f1")), number(first.get("radar_f1"))
        for field in ("TP", "TN", "FP", "FN", "precision", "recall", "f1", "accuracy"):
            row["overlap_" + field] = number(first.get("overlap_" + field))
        fs = (row["lte_f1"], row["radar_f1"])
        row["macro_f1"] = sum(fs)/2 if all(f is not None for f in fs) else None
        route = meta.get("routes", {}).get(f"{row['batch']}:{row['mode']}", {})
        evaluation = route.get("evaluation", {})
        row["evaluation_split"] = evaluation.get("split", meta.get("experiment", {}).get("protocol", {}).get("evaluation_split", "unknown"))
        row["evaluated_images"] = first.get("evaluated_images", evaluation.get("evaluated_images", ""))
        row["operating_confidence"] = evaluation.get("operating_confidence", "")
    return result


def load_run(run, baseline="torch_cuda_fp32"):
    if (run / "data/metadata.json").is_file() or (run / "metadata.json").is_file():
        from .results_store import load_data
        return load_data(run, baseline)
    models, source = {}, []
    for path in sorted(run.glob("*/model.json")):
        meta_bytes = read_bytes(path, run.parent)
        result_bytes = read_bytes(path.parent / "results.csv", run.parent)
        meta, raw = json.loads(meta_bytes), csv_rows(result_bytes)
        if meta["model_id"] != path.parent.name:
            raise ValueError(f"Model identifier differs from directory: {path}")
        models[meta["model_id"]] = {"meta": meta, "raw": raw, "rows": enrich(meta, raw, baseline)}
        source.extend({"path": f"{path.parent.name}/{name}", "sha256_before_sanitizing": sha(data)}
                      for name, data in (("model.json", meta_bytes), ("results.csv", result_bytes)))
    if not models:
        raise FileNotFoundError(f"No per-model model.json/results.csv pairs in {run}")
    first = next(iter(models.values()))["meta"]
    order = [m for m in first["experiment"]["model_order"] if m in models]
    order.extend(m for m in models if m not in order)
    return {"models": models, "order": order, "run_name": first["experiment"].get("name", run.name),
            "source": source, "baseline": baseline}


def cohorts(rows):
    values = defaultdict(list)
    for row in rows:
        values[(row.get("timing_signature", ""), row.get("quality_signature", ""))].append(row)
    return values


def route_records(bundle):
    result = []
    for model in bundle["order"]:
        meta = bundle["models"][model]["meta"]
        for batch in meta["experiment"]["batches"]:
            for mode in meta["experiment"]["modes"]:
                entry = meta.get("routes", {}).get(f"{batch}:{mode}", {})
                result.append({"model": model, "batch": batch, "mode": mode,
                    "status": entry.get("status", "missing"), "phase": entry.get("phase", ""),
                    "error": sanitize(entry.get("error", "")), "execution_kind": entry.get("execution_kind", ""),
                    "started_at": entry.get("started_at", ""), "finished_at": entry.get("finished_at", ""),
                    "elapsed_seconds": entry.get("elapsed_seconds", "")})
    return result


def portable_source(bundle, output, dataset_manifest=None):
    for model, item in bundle["models"].items():
        directory = output / "source" / model
        write_json(directory / "model.json", sanitize(item["meta"]))
        write_csv(directory / "results.csv", sanitize(item["raw"]))
    records = []
    for source in bundle["source"]:
        exported = output / "source" / source["path"]
        records.append({**source, "sha256_published": sha(exported.read_bytes())})
    manifest = {"schema": 1, "run_name": bundle["run_name"], "files": records,
                "note": "Private home-directory prefixes are replaced with <USER_HOME>. Metrics are unchanged."}
    if dataset_manifest and dataset_manifest.is_file():
        original = dataset_manifest.read_bytes()
        data = json.loads(original)
        expected = {i["meta"].get("dataset", {}).get("records_sha256") for i in bundle["models"].values()}
        if data.get("records_sha256") not in expected or len(expected) != 1:
            raise ValueError("Dataset manifest does not match recorded experiment membership")
        write_json(output / "source/dataset.json", sanitize(data))
        manifest["dataset_manifest_original_sha256"] = sha(original)
    modules = ("rt_report.py", "report_plots.py", "compact.py")
    manifest["analysis_code_sha256"] = {name: sha((Path(__file__).parent/name).read_bytes()) for name in modules}
    write_json(output / "source_manifest.json", manifest)


def fmt(value, places=4):
    value = number(value)
    return "—" if value is None else f"{value:.{places}f}"


def image_link(files, output, prefix=""):
    png = next((p for p in files if p.suffix == ".png"), files[0] if files else None)
    if png is None:
        return ""
    label = "![Policy comparison]" if png.suffix == ".png" else "[Policy comparison (PDF)]"
    return f"{label}({prefix}{png.relative_to(output).as_posix()})"


def observations(rows):
    lookup = {(r["model"], int(r["batch"]), r["scope"], r["mode"]): r for r in rows}
    models = list(dict.fromkeys(r["model"] for r in rows))
    pairs = [(lookup.get((m,1,"model_compute","torch_cuda_fp32")),
              lookup.get((m,1,"model_compute","trt_fp16"))) for m in models]
    pairs = [(a,b) for a,b in pairs if a and b]
    text = []
    if pairs:
        reductions = [100*(1-b["inference_time_per_frame_s"]/a["inference_time_per_frame_s"]) for a,b in pairs]
        f1 = [abs(b["overlap_f1"]-a["overlap_f1"]) for a,b in pairs if a["overlap_f1"] is not None and b["overlap_f1"] is not None]
        maps = [abs(float(b["map50"])-float(a["map50"])) for a,b in pairs]
        text.append(f"- **FP16 model computation:** direct TensorRT FP16 reduces time per frame by "
                    f"{min(reductions):.1f}–{max(reductions):.1f}% versus PyTorch FP32 across {len(pairs)} models at batch 1. "
                    f"The largest absolute mAP@0.5 change is {max(maps):.5f}"+
                    (f", and the largest detection-F1 change is {max(f1):.5f}." if f1 else "."))
    quantized = []
    for model in models:
        a=lookup.get((model,1,"model_compute","torch_cuda_fp32"))
        b=lookup.get((model,1,"model_compute","trt_int8"))
        if a and b:
            quantized.append((float(a["map50"])-float(b["map50"]), model, a,b))
    if quantized:
        _, model,a,b=max(quantized, key=lambda x:x[0])
        text.append(f"- **INT8 quality depends on the model:** {model_label(model)} shows the largest batch-1 "
                    f"mAP@0.5 reduction among direct INT8 routes: {float(a['map50']):.4f} → {float(b['map50']):.4f}. "
                    f"Overlap F1 changes from {fmt(a['overlap_f1'])} to {fmt(b['overlap_f1'])}. "
                    "The detailed reports also show batch-dependent quality changes.")
    png_pairs = []
    for model in models:
        a=lookup.get((model,1,"image_to_detection","torch_cuda_fp32"))
        b=lookup.get((model,1,"image_to_detection","trt_fp16"))
        if a and b:
            png_pairs.append((model,a,b))
    if png_pairs:
        improved=sum(b["inference_time_per_frame_s"] < a["inference_time_per_frame_s"] for _,a,b in png_pairs)
        text.append(f"- **Full image inference has a different result:** with PNG decoding and preprocessing included, "
                    f"direct TensorRT FP16 reduces time per frame in {improved} of {len(png_pairs)} batch-1 comparisons. "
                    "The forward-pass gain should therefore be read together with the full-pipeline measurements.")
    return text






def generate(run, output, formats=("png", "pdf"), dpi=180, baseline="torch_cuda_fp32", scope="all",
             overview_scope="model_compute", dataset_manifest=None):
    from .results_store import export_data, included, seal
    from .report_text import write_reports
    bundle = load_run(run, baseline)
    rows = [r for m in bundle["order"] for r in bundle["models"][m]["rows"] if included(r,bundle["report_scope"])]
    if rows and any(r.get("overlap_f1") is None for r in rows):
        raise ValueError("Overlap F1 is missing from this run. Historical box F1 cannot be converted to overlap F1. Keep the archived report; analyze a new run with recorded overlap counts.")
    output.mkdir(parents=True, exist_ok=True)
    data = output / "data"
    export_data(bundle, data, dataset_manifest)
    raw = [r for m in bundle["order"] for r in bundle["models"][m]["raw"]]
    routes = route_records(bundle)
    for route in routes:
        route["in_scope"] = included(route,bundle["report_scope"])
    write_csv(data / "summary.csv", rows)
    write_csv(data / "routes.csv", routes)
    seal(data)
    charts, files = {}, []
    target_scopes = SCOPES if scope == "all" else (scope,)
    for model in bundle["order"]:
        for timing_scope in target_scopes:
            selected = [r for r in rows if r["model"]==model and r["scope"] == timing_scope]
            groups = cohorts(selected)
            for signature, group in groups.items():
                suffix = "_"+signature[0][:8]+"_"+signature[1][:8] if len(groups)>1 else ""
                made = model_chart(model, group, output / "figures", timing_scope, formats, dpi, suffix)
                files.extend(made)
                charts.setdefault((model, timing_scope), []).append(made)
        print(f"  Report: {model}", flush=True)
    # Each model/batch panel must have only one protocol; model hashes naturally differ between models.
    for model in bundle["order"]:
        selected = [r for r in rows if r["model"]==model and r["scope"]==overview_scope and int(r["batch"])==1]
        if len(cohorts(selected)) > 1:
            raise ValueError(f"Cannot combine protocol cohorts in overview for {model}")
    made = overview(rows, bundle["order"], output / "figures", overview_scope, 1, formats, dpi)
    files.extend(made)
    overview_groups = [[p for p in made if p.stem == stem] for stem in dict.fromkeys(p.stem for p in made)]
    files.extend(coverage_chart([r for r in routes if r["in_scope"]], output / "figures", formats, dpi))
    write_reports(bundle, rows, routes, charts, overview_groups, output)
    return {"run_name": bundle["run_name"], "models": len(bundle["models"]), "comparisons": len(rows),
            "raw_rows": len(raw), "routes": sum(r["in_scope"] for r in routes), "figures": len(files)}


def update_readme(project, output):
    path = project / "README.md"
    text = path.read_text(encoding="utf-8")
    relative = output.relative_to(project).as_posix()
    summary = (output / "SUMMARY.md").read_text(encoding="utf-8")
    summary = re.sub(r"\]\((?!https?://)([^)]+)\)", lambda m: "]("+relative+"/"+m.group(1)+")", summary)
    if "\n## Results\n" in text:
        start = text.index("\n## Results\n")+1
        end = text.find("\n## ", start+3)
        text = text[:start]+summary+text[end if end>=0 else len(text):]
    else:
        location = text.find("\n## ")
        location = location if location>=0 else len(text)
        text = text[:location]+"\n"+summary+text[location:]
    path.write_text(text, encoding="utf-8")
