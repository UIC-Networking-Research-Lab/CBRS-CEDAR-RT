"""One portable, hash-checked store for final measurements and earlier attempts."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil

CORE_FILES = ("metadata.json", "rounds.csv", "dataset.json", "history.jsonl")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_directory(path):
    path = Path(path).resolve()
    return path / "data" if (path / "data/metadata.json").is_file() else path


def verify(path):
    directory = data_directory(path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 2:
        raise ValueError("Unsupported consolidated result schema")
    for record in manifest["files"]:
        path = (directory / record["path"]).resolve()
        if not path.is_relative_to(directory) or path.parent != directory:
            raise ValueError("Result manifest path escapes the data directory")
        if digest(path) != record["sha256"]:
            raise ValueError(f"Result checksum mismatch: {path.name}")
    required = {"metadata.json", "rounds.csv"}
    if not required.issubset({r["path"] for r in manifest["files"]}):
        raise ValueError("Result manifest omits required source data")
    return manifest


def load_data(path, baseline="torch_cuda_fp32"):
    from .rt_report import enrich
    directory = data_directory(path)
    verify(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != 2:
        raise ValueError("Unsupported consolidated metadata schema")
    with (directory / "rounds.csv").open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in raw:
        if row["model"] not in metadata["models"]:
            raise ValueError(f"Unknown model in rounds.csv: {row['model']}")
        groups[row["model"]].append(row)
    models = {}
    for model in metadata["model_order"]:
        meta = metadata["models"][model]
        if meta["model_id"] != model:
            raise ValueError(f"Metadata model identifier mismatch: {model}")
        models[model] = {"meta": meta, "raw": groups[model], "rows": enrich(meta, groups[model], baseline)}
    return {"models": models, "order": metadata["model_order"], "run_name": metadata["run_name"],
            "source": metadata.get("source_evidence", []), "baseline": baseline,
            "report_scope": metadata["report_scope"], "data_directory": directory}


def included(row, scope):
    return (row["model"] in scope["models"] and int(row["batch"]) in scope["batches"]
            and row["mode"] in scope["modes"])


def export_data(bundle, directory, dataset_manifest=None):
    """Copy immutable inputs, or consolidate a newly recorded compact run once."""
    from .rt_report import sanitize, write_csv, write_json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = bundle.get("data_directory")
    if source:
        if source.resolve() != directory.resolve():
            for name in CORE_FILES:
                if (source / name).is_file():
                    shutil.copy2(source / name, directory / name)
        return
    first = bundle["models"][bundle["order"][0]]["meta"]
    existing = directory / "metadata.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8"))["run_name"] != bundle["run_name"]:
        raise ValueError("This data store belongs to another experiment; publish the new run to a different directory")
    scope = bundle.get("report_scope", {"models": bundle["order"],
        "batches": first["experiment"]["batches"], "modes": first["experiment"]["modes"]})
    bundle["report_scope"] = scope
    metadata = {"schema": 2, "run_name": bundle["run_name"], "model_order": bundle["order"],
                "report_scope": scope, "source_evidence": bundle["source"],
                "models": {model: sanitize(bundle["models"][model]["meta"]) for model in bundle["order"]}}
    write_json(directory / "metadata.json", metadata)
    write_csv(directory / "rounds.csv", sanitize([r for model in bundle["order"] for r in bundle["models"][model]["raw"]]))
    if dataset_manifest and Path(dataset_manifest).is_file():
        data = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
        expected = {v["meta"].get("dataset", {}).get("records_sha256") for v in bundle["models"].values()}
        if len(expected) != 1 or data.get("records_sha256") not in expected:
            raise ValueError("Dataset manifest differs from experiment membership")
        write_json(directory / "dataset.json", sanitize(data))
    if not (directory / "history.jsonl").exists():
        (directory / "history.jsonl").write_text("", encoding="utf-8")


def seal(directory, audit=None):
    """Hash immutable inputs and derived tables after a complete analysis write."""
    from .rt_report import write_json
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    previous = directory / "manifest.json"
    if audit is None and previous.exists():
        audit = json.loads(previous.read_text(encoding="utf-8")).get("audit", {})
    names = [*CORE_FILES, "summary.csv", "routes.csv"]
    files = [{"path": name, "sha256": digest(directory/name), "bytes": (directory/name).stat().st_size}
             for name in names if (directory/name).is_file()]
    modules = ("results_store.py", "rt_report.py", "report_text.py", "report_plots.py", "compact.py")
    write_json(directory / "manifest.json", {"schema": 2, "run_name": metadata["run_name"],
        "scope": metadata["report_scope"], "files": files, "audit": audit or {},
        "analysis_code_sha256": {name: digest(Path(__file__).parent/name) for name in modules},
        "note": "Original numeric values are retained. Private home prefixes are sanitized. "
                "Earlier attempts are provenance, not additional repetitions of final measurements."})


def table(path, name="summary"):
    """Read a combined table without importing plotting or inference dependencies."""
    if name not in ("summary", "rounds", "routes"):
        raise ValueError("Choose summary, rounds or routes")
    directory = data_directory(path)
    verify(directory)
    with (directory / f"{name}.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))
