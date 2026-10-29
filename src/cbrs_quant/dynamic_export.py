"""Export a dynamic-batch ONNX without changing the fixed-batch experiment path."""

from pathlib import Path

import torch

from .data import load_png, to_tensor
from .export import insert_quantizers
from .io import digest, environment, fingerprint, reserve_output, sidecar, write_json
from .models import load_model, prepare_export, require_cuda


@torch.inference_mode()
def export_dynamic(model_id, weights, output, precision="fp32", device_name="cpu",
                   dataset=None, samples=512, seed=42, progress=None):
    import onnx

    if precision not in ("fp32", "fp16", "int8") or device_name not in ("cpu", "cuda"):
        raise ValueError("Unsupported precision or device")
    if precision == "fp16" and device_name != "cuda":
        raise ValueError("FP16 export requires CUDA")
    if (precision == "int8") != (dataset is not None):
        raise ValueError("INT8 requires calibration data; floating exports do not")
    output = reserve_output(output)
    device = require_cuda() if device_name == "cuda" else torch.device("cpu")
    model, meta = load_model(model_id, Path(weights))
    model.to(device)
    prepare_export(model)
    calibration = None
    if dataset is not None:
        # Calibration PNGs must use the training rendering profile and library versions.
        selected = dataset.calibration(samples, seed)
        quantizers = insert_quantizers(model)
        for index, record in enumerate(selected, 1):
            model(to_tensor([load_png(dataset.path(record["image"]))], device, torch.float32))
            if progress:
                progress(index, len(selected))
        for _, quantizer in quantizers:
            quantizer.freeze()
        calibration = {
            "algorithm": "symmetric_absmax_ptq", "weight_axis": 0, "activation_axis": None,
            "range": [-128, 127], "scale_divisor": 127, "seed": seed, "count": len(selected),
            "split": "train", "selection_sha256": fingerprint(selected), "records": selected,
            "dataset_sha256": dataset.manifest["records_sha256"],
            "profile_sha256": dataset.manifest["profile_sha256"],
            "policy": "fused_conv_bn; backbone_and_neck_int8; complete_head_float",
            "layers": [{"name": name, "input_scale": q.input_scale, "output_scale": q.output_scale}
                       for name, q in quantizers],
        }
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model.to(dtype=dtype).eval()
    example = torch.zeros((1, 3, 640, 640), device=device, dtype=dtype)
    eager = model(example)
    if not torch.isfinite(eager).all():
        raise ValueError("Model produced non-finite output before export")
    temporary = output.with_name(output.name + ".tmp.onnx")
    try:
        torch.onnx.export(model, example, temporary, input_names=["images"], output_names=["predictions"],
                          dynamic_axes={"images": {0: "batch"}, "predictions": {0: "batch"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)
        graph = onnx.load(str(temporary))
        onnx.checker.check_model(graph)
        expected = onnx.TensorProto.FLOAT16 if eager.dtype == torch.float16 else onnx.TensorProto.FLOAT
        if graph.graph.output[0].type.tensor_type.elem_type != expected:
            raise ValueError("ONNX output dtype differs from PyTorch output")
        counts = {op: sum(node.op_type == op for node in graph.graph.node)
                  for op in ("QuantizeLinear", "DequantizeLinear")}
        if precision == "int8" and min(counts.values()) == 0:
            raise ValueError("INT8 export contains no explicit Q/DQ nodes")
        meta.update(kind="onnx", graph_precision="int8_qdq" if precision == "int8" else precision,
                    batch=None, batch_dynamic=True, example_batch=1,
                    input_dtype=str(dtype).removeprefix("torch."),
                    output_dtype=str(eager.dtype).removeprefix("torch."),
                    output_shape=["batch", *list(eager.shape[1:])], calibration=calibration,
                    qdq_nodes=counts, opset=17, environment=environment(),
                    validation="onnx.checker; validate runtime parity for target batch sizes")
        temporary.replace(output)
        meta["sha256"] = digest(output)
        write_json(sidecar(output), meta)
    finally:
        temporary.unlink(missing_ok=True)
    return meta
