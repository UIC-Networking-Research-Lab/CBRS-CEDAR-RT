"""Build target-local TensorRT engines; explicit Q/DQ replaces legacy calibrators."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .io import digest, environment, onnx_precision, reserve_output, sidecar, verified_artifact, write_json
from .models import require_cuda


def build_engine(onnx_path: Path, output: Path, workspace_gib: float = 2,
                 precision: str | None = None, optimization_level: int = 3) -> dict:
    import tensorrt as trt
    if int(trt.__version__.split(".")[0]) != 10:
        raise RuntimeError("Use TensorRT 10.x for this release")
    if workspace_gib <= 0:
        raise ValueError("Workspace must be positive")
    if not 0 <= optimization_level <= 5:
        raise ValueError("TensorRT optimization level must be in [0,5]")
    require_cuda()
    source = verified_artifact(onnx_path, "onnx")
    graph_precision = onnx_precision(source)
    precision = precision or ("int8" if graph_precision == "int8_qdq" else "fp16")
    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError("Unsupported engine precision")
    if (precision == "int8") != (graph_precision == "int8_qdq"):
        raise ValueError("INT8 requires a calibrated Q/DQ graph; floating modes require floating graphs")
    if precision == "fp32" and graph_precision != "fp32":
        raise ValueError("FP32 engine requires an FP32 ONNX graph")
    reserve_output(output)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path.resolve())):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parsing failed:\n{errors}")
    config = builder.create_builder_config()
    config.builder_optimization_level = optimization_level
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 1024**3))
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision != "fp32":
        config.set_flag(trt.BuilderFlag.FP16)
    # In explicit-quantization networks the Q/DQ graph controls INT8 placement.
    # Do not set INT8 or provide a legacy calibrator; unquantized layers may use FP16.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed; inspect the builder log")
    build_seconds = time.perf_counter() - started
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Built engine could not be deserialized")
    inspector = engine.create_engine_inspector()
    inspection = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    layers = json.loads(inspection)
    # TensorRT versions vary in field names; preserve the complete inspector report.
    # A token count is evidence, not a FLOP-weighted precision coverage claim.
    int8_mentions = inspection.lower().count("int8")
    if precision == "int8" and int8_mentions == 0:
        raise RuntimeError("Q/DQ engine reports no INT8 layers; refusing an INT8 label")
    output.write_bytes(bytes(plan))
    meta = {**source, "kind": "engine", "mode": f"trt_{precision}", "route": "C",
            "precision": precision, "sha256": digest(output),
            "onnx_sha256": source["sha256"], "environment": environment(),
            "build_seconds": build_seconds, "workspace_gib": workspace_gib,
            "builder_optimization_level": optimization_level,
            "tf32": False, "float_fallback": "FP16/FP32 as selected by TensorRT",
            "int8_inspector_mentions": int8_mentions,
            "precision_coverage": "Inspect layers; no FLOP-weighted claim", "layers": layers}
    write_json(sidecar(output), meta)
    return meta
