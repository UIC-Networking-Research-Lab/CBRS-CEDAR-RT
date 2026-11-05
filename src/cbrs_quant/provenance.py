"""Checkpoint precision and experiment metadata."""

from datetime import datetime, timezone


def precision_notes(storage):
    """Describe the observed checkpoint, rather than assuming dtype by model name."""
    types = set(storage.get("floating_tensor_dtypes", {}))
    if types == {"float16"}:
        origin = "FP32 ONNX is exported from FP16-stored checkpoint values; lost precision cannot be restored."
    elif types == {"float32"}:
        origin = "FP32 ONNX is exported from FP32-stored checkpoint values."
    else:
        origin = "FP32 ONNX is exported from the checkpoint; see floating_tensor_dtypes for its storage."
    return {**storage, "onnx_fp32_origin": origin,
            "comparison_note": "FP32/FP16 quality is often close; verify evaluation metrics. "
                               "INT8 quality depends on quantization and calibration."}


def checkpoint_info(path):
    """Inspect checkpoint storage; never run a model or change the checkpoint."""
    from collections import Counter
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ema = checkpoint.get("ema")
    model = ema if ema is not None else checkpoint.get("model")
    values = model.values() if isinstance(model, dict) else model.state_dict().values()
    counts = Counter(str(value.dtype).removeprefix("torch.") for value in values
                     if isinstance(value, torch.Tensor) and value.is_floating_point())
    # Ultralytics checkpoints here store FP16; CEDAR stores FP32. Check the actual tensors.
    return precision_notes({"floating_tensor_dtypes": dict(counts),
                            "selected_state": "ema" if ema is not None else "model",
                            "fp32_inference_note": "FP32 computation uses values loaded from this checkpoint; "
                                                   "converting FP16-stored values to FP32 cannot restore lost precision."})


def experiment_info(config, name, source_run):
    return {"name": name, "created_at": datetime.now(timezone.utc).isoformat(),
            "source_run": str(source_run) if source_run else None,
            "original_created_at": config.get("created_at"), "model_order": config["model_order"],
            "batches": config["batches"], "modes": config["modes"], "threads": config["threads"],
            "smoke": config["smoke"], "protocol": config["protocol"],
            "data": config["data"], "manifest": config["manifest"],
            "manifest_sha256": config["manifest_sha256"],
            "minimum_free_gib": config.get("minimum_free_gib", 6),
            "maximum_worker_gib": config.get("maximum_worker_gib", 8)}
