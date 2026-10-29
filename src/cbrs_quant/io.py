"""Small, deterministic artifact and provenance helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: object) -> None:
    """Replace a report atomically; never leave a partial JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def reserve_output(path: str | Path) -> Path:
    """Artifacts are immutable by default; choose a new name to rebuild."""
    path = Path(path)
    if path.exists() or sidecar(path).exists():
        raise FileExistsError(f"Artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sidecar(path: str | Path) -> Path:
    return Path(str(path) + ".json")


def environment() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "ultralytics", "numpy", "Pillow", "onnx",
                 "tensorrt", "onnxruntime", "onnxruntime-gpu", "matplotlib", "scipy", "cbrs-yolo-gpu-quant"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "machine": platform.machine(), "packages": packages,
              "expected_container": "dustynv/pytorch:2.7-r36.4.0-cu128-24.04"}
    result["nvidia_tf32_override"] = os.environ.get("NVIDIA_TF32_OVERRIDE")
    from . import __version__
    result["source_version"] = __version__
    l4t = Path("/etc/nv_tegra_release")
    result["l4t"] = l4t.read_text().strip() if l4t.exists() else None
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda)
        result.update(torch_threads=torch.get_num_threads(),
                      torch_interop_threads=torch.get_num_interop_threads())
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            result.update(gpu=device.name, gpu_memory_bytes=device.total_memory,
                          compute_capability=list(torch.cuda.get_device_capability(0)))
    except ImportError:
        result["cuda_available"] = False
    try:
        import tensorrt
        result["packages"]["tensorrt"] = tensorrt.__version__
    except (ImportError, OSError) as exc:
        result["tensorrt_import_error"] = str(exc)
    try:
        import onnxruntime as ort
        result["ort"] = {"version": ort.__version__, "available_providers": ort.get_available_providers()}
    except (ImportError, OSError) as exc:
        result["ort"] = None
        result["ort_import_error"] = str(exc)
    return result


def verified_artifact(path: str | Path, kind: str) -> dict:
    meta = read_json(sidecar(path))
    if meta.get("kind") != kind or meta.get("sha256") != digest(path):
        raise ValueError(f"Invalid or stale {kind} manifest: {path}")
    return meta


def onnx_precision(meta: dict) -> str:
    """Read current manifests and the original FP32/QDQ export manifests."""
    if "graph_precision" in meta:
        return meta["graph_precision"]
    return "int8_qdq" if meta.get("qdq_nodes", {}).get("QuantizeLinear", 0) else "fp32"
