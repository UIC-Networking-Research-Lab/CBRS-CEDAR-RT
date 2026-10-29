"""Load trusted local checkpoints; never download or guess a model architecture."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn

from .io import digest


class Detector(nn.Module):
    """Expose one decoded tensor; retain the checkpoint's detection branch."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images):
        prediction = self.model(images)
        return prediction[0] if isinstance(prediction, (tuple, list)) else prediction


class CEDARExportHead(nn.Module):
    """Preserve CEDAR decoding while making anchor FP32 explicit for ONNX.

    Tracing integer arange + 0.5 can otherwise promote ONNX anchors to FP64,
    despite eager PyTorch using FP32. The archived source stays unchanged.
    """

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, features):
        head = self.head
        batch, _, height, width = features.shape
        raw = head.cv2(features).reshape(batch, head.no_box, height*width)
        scores = head.cv3(features).reshape(batch, head.nc, height*width).sigmoid()
        f, t = head.reg_max_freq, head.reg_max_time
        left = head._dfl(raw[:, :f], f) * head.stride_freq
        right = head._dfl(raw[:, f:2*f], f) * head.stride_freq
        top = head._dfl(raw[:, 2*f:2*f+t], t) * head.stride_time
        bottom = head._dfl(raw[:, 2*f+t:], t) * head.stride_time
        gy = (torch.arange(height, device=features.device, dtype=torch.float32)+.5)*head.stride_time
        gx = (torch.arange(width, device=features.device, dtype=torch.float32)+.5)*head.stride_freq
        cy, cx = torch.meshgrid(gy, gx, indexing="ij")
        cx, cy = cx.reshape(-1), cy.reshape(-1)
        boxes = torch.stack((cx-left, cy-top, cx+right, cy+bottom), dim=1)
        return torch.cat((boxes, scores), dim=1)


def load_model(model_id: str, weights: Path) -> tuple[Detector, dict]:
    if not weights.is_file():
        raise FileNotFoundError(weights)
    cedar = re.fullmatch(r"sig_yolo_(v8)([nsx])_f(\d+)t(\d+)_c(\d+)", model_id)
    yolo = re.fullmatch(r"cbrs_yolo(8|11|26)([nsx])", model_id)
    if cedar:
        from .vendor.asym_yolo import build_asym_yolo
        # PyTorch checkpoints contain pickle. Only load weights you trust.
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        variant, size, freq, time, coverage = cedar.groups()
        freq, time = int(freq), int(time)
        cov = int(coverage) / 100
        config = dict(variant=variant, size=size, nc=2, fm_freq=freq, fm_time=time,
                      reg_max_freq=max(1, round(freq*cov)),
                      reg_max_time=max(1, round(time*cov)), input_w=640, input_h=640)
        for key in ("variant", "size", "nc", "fm_freq", "fm_time",
                    "reg_max_freq", "reg_max_time"):
            if key in ckpt and ckpt[key] != config[key]:
                raise ValueError(f"Checkpoint {key}={ckpt[key]} disagrees with {model_id}")
        if config["nc"] != 2 or ckpt.get("img_size", 640) != 640:
            raise ValueError("Only two-class 640x640 CBRS checkpoints are supported")
        model = build_asym_yolo(**config)
        state = ckpt.get("ema")
        if state is None:
            state = ckpt["model"]
        model.load_state_dict(state, strict=True)
        layout = "xyxy_scores"
    elif yolo:
        from ultralytics import YOLO
        checkpoint = YOLO(str(weights), task="detect")
        model = checkpoint.model
        identity = model.yaml.get("yaml_file") or checkpoint.ckpt.get("train_args", {}).get("model")
        yaml_file = Path(identity or "").stem
        expected_yaml = f"yolo{'v' if yolo.group(1) == '8' else ''}{yolo.group(1)}{yolo.group(2)}"
        if yaml_file != expected_yaml or model.yaml.get("scale", yolo.group(2)) != yolo.group(2):
            raise ValueError(f"Checkpoint architecture {yaml_file} disagrees with {model_id}")
        if [model.names.get(i) for i in range(2)] != ["LTE", "Radar"]:
            raise ValueError("Checkpoint class order must be [LTE, Radar]")
        head = model.model[-1]
        if head.nc != 2:
            raise ValueError(f"Expected two classes, found {head.nc}")
        end2end = bool(getattr(head, "end2end", False))
        if end2end != (yolo.group(1) == "26"):
            raise ValueError("Model identifier and checkpoint detection branch disagree")
        if end2end:
            # Native box ranking (top-k) stays inside the detector. No feature codec.
            head.fuse()  # Remove unused one2many training branch in every backend.
        layout = "end2end" if end2end else "xywh_scores"
        config = {"family": yolo.group(1), "size": yolo.group(2), "end2end": end2end}
    else:
        raise ValueError(f"Unsupported model identifier: {model_id}")
    model = Detector(model.float().eval())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"model_id": model_id, "weights_sha256": digest(weights),
                   "layout": layout, "classes": ["LTE", "Radar"], "image_size": [640, 640],
                   "architecture": config}


def prepare_export(model: Detector) -> None:
    """Use native decoded heads without adding NMS to the ONNX graph."""
    if hasattr(model.model, "head") and not isinstance(model.model.head, CEDARExportHead):
        model.model.head = CEDARExportHead(model.model.head).eval()
    for module in model.modules():
        if hasattr(module, "export") and hasattr(module, "stride"):
            module.export = True
            module.format = "onnx"
            module.dynamic = False


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Use the Jetson PyTorch container; no CPU fallback.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda:0")
