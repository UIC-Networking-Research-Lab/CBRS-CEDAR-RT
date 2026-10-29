"""Static ONNX export and symmetric INT8 PTQ using explicit Q/DQ.

WARNING: calibration must use PNGs from the training rendering profile. Changing
spectral parameters or renderer versions can invalidate both weights and scales.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .data import Dataset, load_png, to_tensor
from .io import digest, environment, fingerprint, reserve_output, sidecar, write_json
from .models import load_model, prepare_export, require_cuda


class QDQConv(nn.Module):
    """Per-channel signed weights and per-tensor signed activations; zero point 0."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        scale = conv.weight.detach().abs().flatten(1).amax(1).clamp_min(1e-8) / 127
        self.register_buffer("weight_scale", scale)
        self.register_buffer("weight_zero", torch.zeros_like(scale, dtype=torch.int32))
        self.register_buffer("input_amax", torch.zeros((), device=conv.weight.device))
        self.register_buffer("output_amax", torch.zeros((), device=conv.weight.device))
        self.calibrating = True
        self.input_scale = self.output_scale = 1.0

    def forward(self, images):
        if self.calibrating:
            self.input_amax.copy_(torch.maximum(self.input_amax, images.detach().abs().amax()))
            result = self.conv(images)
            self.output_amax.copy_(torch.maximum(self.output_amax, result.detach().abs().amax()))
            return result
        images = torch.fake_quantize_per_tensor_affine(images, self.input_scale, 0, -128, 127)
        weight = torch.fake_quantize_per_channel_affine(
            self.conv.weight, self.weight_scale, self.weight_zero, 0, -128, 127)
        result = self.conv._conv_forward(images, weight, self.conv.bias)
        return torch.fake_quantize_per_tensor_affine(result, self.output_scale, 0, -128, 127)

    def freeze(self):
        if not torch.isfinite(self.input_amax) or not torch.isfinite(self.output_amax):
            raise ValueError("Non-finite calibration activations")
        self.input_scale = max(float(self.input_amax), 1e-8) / 127
        self.output_scale = max(float(self.output_amax), 1e-8) / 127
        self.calibrating = False


def insert_quantizers(model: nn.Module) -> list[tuple[str, QDQConv]]:
    """Fuse Conv+BN before observing; keep the complete detection head floating."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    for module in list(model.modules()):
        conv, bn = getattr(module, "conv", None), getattr(module, "bn", None)
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            module.conv = fuse_conv_bn_eval(conv, bn)
            module.bn = nn.Identity()
    # Wrapper.model is CEDAR or an Ultralytics DetectionModel.
    base = model.model
    head_prefix = "model.head" if hasattr(base, "head") else f"model.model.{len(base.model)-1}"
    selected = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d) and not name.startswith(head_prefix + "."):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            quantizer = QDQConv(module)
            setattr(parent, child_name, quantizer)
            selected.append((name, quantizer))
    if not selected:
        raise ValueError("No eligible convolutions were found for PTQ")
    return selected


@torch.inference_mode()
def export_onnx(model_id: str, weights: Path, output: Path, batch: int = 1,
                dataset: Dataset | None = None, samples: int = 512, seed: int = 42,
                precision: str = "fp32", device_name: str = "cuda", progress=None) -> dict:
    import onnx
    if batch < 1:
        raise ValueError("Batch must be positive")
    if precision not in ("fp32", "fp16") or device_name not in ("cuda", "cpu"):
        raise ValueError("Unsupported export precision or device")
    if (dataset is not None and precision != "fp32") or (precision == "fp16" and device_name == "cpu"):
        raise ValueError("Q/DQ calibration requires FP32; FP16 export requires CUDA")
    reserve_output(output)
    device = require_cuda() if device_name == "cuda" else torch.device("cpu")
    model, meta = load_model(model_id, weights)
    model.to(device)
    prepare_export(model)
    calibration = None
    if dataset is not None:
        selected = dataset.calibration(samples, seed)
        quantizers = insert_quantizers(model)
        for index, record in enumerate(selected):
            model(to_tensor([load_png(dataset.path(record["image"]))], device, torch.float32))
            if progress:
                progress({"completed": index+1, "total": len(selected)})
        for _, quantizer in quantizers:
            quantizer.freeze()
        calibration = {
            "algorithm": "symmetric_absmax_ptq", "weight_axis": 0, "activation_axis": None,
            "range": [-128, 127], "scale_divisor": 127, "seed": seed, "count": len(selected),
            "split": "train", "selection_sha256": fingerprint(selected), "records": selected,
            "dataset_sha256": dataset.manifest["records_sha256"],
            "profile_sha256": dataset.manifest["profile_sha256"],
            "policy": "fused_conv_bn; backbone_and_neck_int8; complete_head_float",
            "layers": [{"name": name, "input_scale": q.input_scale,
                        "output_scale": q.output_scale} for name, q in quantizers],
        }
    if progress:
        progress({"measurement": "exporting_onnx"})
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model.to(dtype=dtype).eval()
    dummy = torch.zeros((batch, 3, 640, 640), device=device, dtype=dtype)
    eager = model(dummy)
    if not torch.isfinite(eager).all():
        raise ValueError("Model produced non-finite output before export")
    temporary = output.with_name(output.name + ".tmp.onnx")
    try:
        # Legacy exporter is intentional: torch 2.7 + fake_quant emits standard Q/DQ.
        torch.onnx.export(model, dummy, temporary, input_names=["images"],
                          output_names=["predictions"], opset_version=17,
                          do_constant_folding=True, dynamo=False)
        graph = onnx.load(str(temporary))
        onnx.checker.check_model(graph)
        actual_output_type = graph.graph.output[0].type.tensor_type.elem_type
        expected_output_type = onnx.TensorProto.FLOAT16 if eager.dtype == torch.float16 else onnx.TensorProto.FLOAT
        if actual_output_type != expected_output_type:
            raise ValueError("ONNX output dtype differs from eager output; refusing an invalid export")
        counts = {op: sum(n.op_type == op for n in graph.graph.node)
                  for op in ("QuantizeLinear", "DequantizeLinear")}
        if dataset is not None and min(counts.values()) == 0:
            raise ValueError("INT8 export did not contain explicit Q/DQ nodes")
        meta.update(kind="onnx", graph_precision="int8_qdq" if dataset is not None else precision,
                    batch=batch, input_dtype=str(dtype).removeprefix("torch."),
                    output_dtype=str(eager.dtype).removeprefix("torch."), output_shape=list(eager.shape),
                    calibration=calibration, qdq_nodes=counts, opset=17,
                    environment=environment(), validation="onnx.checker; GPU parity still required")
        temporary.replace(output)
        meta["sha256"] = digest(output)
        write_json(sidecar(output), meta)
    finally:
        temporary.unlink(missing_ok=True)
    return meta
