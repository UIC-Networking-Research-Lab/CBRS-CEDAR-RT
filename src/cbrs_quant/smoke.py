"""Real CPU/GPU execution checks, with optional checkpoint parity."""

from __future__ import annotations

import torch
import numpy as np

from .data import load_png, to_tensor
from .io import digest, environment
from .postprocess import detections
from .runtime import TorchBackend, synchronize
from .metrics import box_iou


def end2end_parity(actual, expected, min_iou=.95, max_score_error=.05):
    """Match final detection boxes independent of detector-head ranking order.

    YOLO26 selects its final boxes with top-k. This is ordinary local detection,
    not feature compression, a codec, network transmission or MDI partitioning.
    """
    if len(actual) != len(expected):
        return False
    overlaps = box_iou(actual[:, :4], expected[:, :4])
    used = set()
    for i in np.argsort(-actual[:, 4], kind="stable"):
        candidates = [j for j in range(len(expected)) if j not in used
                      and actual[i, 5] == expected[j, 5] and overlaps[i, j] >= min_iou
                      and abs(actual[i, 4]-expected[j, 4]) <= max_score_error]
        if not candidates:
            return False
        used.add(max(candidates, key=lambda j: overlaps[i, j]))
    return True


@torch.inference_mode()
def smoke(backend, paths, reference_weights=None, max_relative_l2=.05,
          confidence=.25, iou=.45, max_det=300, reference_device=None,
          reference_batch=None):
    if max_relative_l2 <= 0:
        raise ValueError("Relative L2 threshold must be positive")
    reference = None
    if reference_weights is not None:
        if digest(reference_weights) != backend.meta["weights_sha256"]:
            raise ValueError("Parity reference must use the same checkpoint")
        target = reference_device or backend.device.type
        if target not in ("cpu", "cuda") or reference_batch not in (None, 1, backend.batch):
            raise ValueError("Reference device must be cpu/cuda; batch must be 1 or backend batch")
        reference_mode = "torch_cpu_fp32" if target == "cpu" else "torch_cuda_fp32"
        reference = TorchBackend(backend.meta["model_id"], reference_weights, reference_mode,
                                 reference_batch or backend.batch)
    records = []
    for path in paths:
        images = [load_png(path)] * backend.batch
        tensor = to_tensor(images, backend.device, backend.dtype)
        raw = backend(tensor)
        if not torch.isfinite(raw).all():
            raise ValueError(f"Non-finite output: {path}")
        det = detections(raw, backend.meta["layout"], confidence, iou, max_det)
        record = {"image": path.name, "shape": list(raw.shape),
                  "detections": [len(d) for d in det]}
        if reference is not None:
            expected = reference(to_tensor(images[:reference.batch], reference.device, torch.float32))
            # Repeated identical PNGs permit a batch-1 CPU reference without a
            # second GPU model. The tested backend still executes its full batch.
            if reference.batch == 1 and backend.batch != 1:
                expected = expected.expand(backend.batch, *expected.shape[1:])
            expected = expected.to(raw.device)
            if expected.shape != raw.shape:
                raise ValueError("Reference and backend output shapes differ")
            errors = (torch.linalg.vector_norm((raw.float()-expected).flatten(1), dim=1)
                      / torch.linalg.vector_norm(expected.flatten(1), dim=1).clamp_min(1e-12))
            error = float(errors.max())
            record["raw_relative_l2"] = error
            record["raw_relative_l2_per_image"] = errors.cpu().tolist()
            if backend.meta["layout"] == "end2end":
                expected_det = detections(expected, "end2end", confidence, iou, max_det)
                passed = all(end2end_parity(a.cpu().numpy(), b.cpu().numpy())
                             for a, b in zip(det, expected_det))
                record["detection_set_parity"] = passed
                if not passed:
                    raise ValueError(f"End-to-end detection set parity failed: {path}")
            elif error > max_relative_l2:
                raise ValueError(f"Raw output parity failed for {path}: {error} > {max_relative_l2}")
        records.append(record)
    synchronize(backend)
    return {"kind": "smoke", "passed": True, "model": backend.meta, "records": records,
            "environment": environment(), "max_relative_l2": max_relative_l2,
            "reference_device": str(reference.device) if reference is not None else None,
            "reference_batch": reference.batch if reference is not None else None,
            "confidence": confidence, "nms_iou": iou, "max_det": max_det,
            "end2end_parity": {"min_iou": .95, "max_score_error": .05},
            "note": "Raw tensor parity is diagnostic; evaluate held-out mAP/recall separately"}
