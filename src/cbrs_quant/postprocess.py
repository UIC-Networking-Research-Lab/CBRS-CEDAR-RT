"""Shared GPU decoding and class-aware NMS; YOLO26 keeps its native end-to-end output."""

from __future__ import annotations

import torch
from torchvision.ops import nms


def detections(raw: torch.Tensor, layout: str, confidence: float = 0.25,
               iou: float = 0.45, max_det: int = 300) -> list[torch.Tensor]:
    if raw.ndim != 3:
        raise ValueError(f"Expected rank-three output, got {tuple(raw.shape)}")
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1 or max_det < 1:
        raise ValueError("Invalid confidence, NMS IoU or max_det")
    raw = raw.float()  # Stable coordinates and NMS for every precision.
    result = []
    for prediction in raw:
        if layout == "end2end":
            if prediction.shape[-1] != 6:
                raise ValueError("End-to-end output must be [B,N,6]")
            det = prediction[prediction[:, 4] > confidence]
            det = det[det[:, 4].argsort(descending=True)[:max_det]]
        else:
            if prediction.shape[0] != 6 or layout not in ("xyxy_scores", "xywh_scores"):
                raise ValueError("Dense output must be [B,6,A] for two classes")
            pred = prediction.T
            scores, classes = pred[:, 4:].max(dim=1)
            selected = scores > confidence
            boxes, scores, classes = pred[selected, :4], scores[selected], classes[selected]
            if layout == "xywh_scores":
                center, half = boxes[:, :2], boxes[:, 2:] / 2
                boxes = torch.cat((center-half, center+half), dim=1)
            if len(scores) > 30000:
                order = scores.argsort(descending=True)[:30000]
                boxes, scores, classes = boxes[order], scores[order], classes[order]
            keep = nms(boxes + classes[:, None] * 7680.0, scores, iou)[:max_det]
            det = torch.cat((boxes[keep], scores[keep, None], classes[keep, None]), dim=1)
        result.append(det)
    return result
