"""Retain immutable PT/ONNX/engine artifacts after experiments, including failed runs."""

from __future__ import annotations

import shutil
from pathlib import Path

from .io import digest, sidecar, verified_artifact, write_json


def archive_models(weights: Path, onnx_paths: list[Path], engines: list[Path], output: Path) -> dict:
    if not onnx_paths:
        raise ValueError("Archive requires at least one ONNX model")
    weight_hash = digest(weights)
    checked = [(p, verified_artifact(p, "onnx")) for p in onnx_paths]
    checked += [(p, verified_artifact(p, "engine")) for p in engines]
    model_ids = {m["model_id"] for _, m in checked}
    if len(model_ids) != 1 or any(m["weights_sha256"] != weight_hash for _, m in checked):
        raise ValueError("Archive artifacts do not share one checkpoint/model identity")
    names = ["checkpoint.pt", "manifest.json"] + [n for p, _ in checked for n in (p.name, sidecar(p).name)]
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError("Archive filenames collide; use distinct precision/batch names")
    onnx_hashes = {m["sha256"] for _, m in checked if m["kind"] == "onnx"}
    if any(m["kind"] == "engine" and m["onnx_sha256"] not in onnx_hashes for _, m in checked):
        raise ValueError("Include the source ONNX for every engine")
    output.mkdir(parents=True, exist_ok=False)
    files = []
    items = [(weights, "checkpoint.pt", "pt")]
    for path, meta in checked:
        items += [(path, path.name, meta["kind"]), (sidecar(path), sidecar(path).name, "metadata")]
    for source, name, kind in items:
        target = output / name
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        actual = digest(target)
        if actual != digest(source):
            raise RuntimeError(f"Copy verification failed: {target}")
        files.append({"path": name, "kind": kind, "sha256": actual, "bytes": target.stat().st_size})
    result = {"schema": 1, "model_id": next(iter(model_ids)), "weights_sha256": weight_hash,
              "files": files, "note": "PT is the original checkpoint; ONNX stores the exported precision/QDQ graph"}
    write_json(output / "manifest.json", result)
    return result
