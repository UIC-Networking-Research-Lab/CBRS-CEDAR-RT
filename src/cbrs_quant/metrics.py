"""CEDAR-MDI quality protocol: IoU-greedy matching and trapezoidal AP.

This is a fixed two-class protocol, not the complete COCO area/crowd evaluator.
"""

from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def box_iou(boxes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    intersect = np.maximum(0, np.minimum(boxes[:, None, 2:], targets[None, :, 2:])
                           - np.maximum(boxes[:, None, :2], targets[None, :, :2])).prod(2)
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    target_areas = np.maximum(0, targets[:, 2:] - targets[:, :2]).prod(1)
    return intersect / (areas[:, None] + target_areas[None, :] - intersect + 1e-9)


def match(pred: np.ndarray, target: np.ndarray, thresholds=IOU_THRESHOLDS) -> np.ndarray:
    """CEDAR-MDI: greedily match by descending IoU, once per prediction/target."""
    correct = np.zeros((len(pred), len(thresholds)), dtype=bool)
    if not len(pred) or not len(target):
        return correct
    overlaps = box_iou(pred[:, :4].astype(np.float32), target[:, 1:5].astype(np.float32))
    same_class = pred[:, 5:6].astype(int) == target[:, 0].astype(int)[None, :]
    for column, threshold in enumerate(thresholds):
        di, ti = np.where((overlaps >= float(threshold)) & same_class)
        pairs = sorted(zip(di, ti, overlaps[di, ti]), key=lambda row: row[2], reverse=True)
        used_pred, used_target = set(), set()
        for d, t, _ in pairs:
            if d not in used_pred and t not in used_target:
                correct[d, column] = True
                used_pred.add(d)
                used_target.add(t)
    return correct


def average_precision(correct: np.ndarray, scores: np.ndarray, count: int) -> float:
    """CEDAR-MDI sentinel endpoints, precision envelope and trapezoidal AP."""
    if count == 0 or not len(scores):
        return 0.0
    ordered = correct[np.argsort(-scores)].astype(float)
    tp, fp = ordered.cumsum(), (1.0 - ordered).cumsum()
    recall = tp / (count + 1e-16)
    precision = tp / (tp + fp + 1e-16)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def frequency_overlap(boxes: np.ndarray, classes: np.ndarray) -> bool:
    """MDI rule: any LTE/radar pair with positive-width frequency intersection."""
    lte, radar = boxes[classes == 0], boxes[classes == 1]
    if not len(lte) or not len(radar):
        return False
    return bool(np.any(np.minimum(lte[:, None, 2], radar[None, :, 2]) >
                       np.maximum(lte[:, None, 0], radar[None, :, 0])))


def overlap_summary(pairs, operating_conf=.25):
    counts = dict(TP=0, TN=0, FP=0, FN=0)
    for prediction, target in pairs:
        selected = prediction[prediction[:, 4] > operating_conf]
        predicted = frequency_overlap(selected[:, :4], selected[:, 5])
        actual = frequency_overlap(target[:, 1:5], target[:, 0])
        counts['TP' if predicted and actual else 'FP' if predicted else 'FN' if actual else 'TN'] += 1
    tp, tn, fp, fn = (counts[k] for k in ('TP', 'TN', 'FP', 'FN'))
    return {**counts, 'precision': tp / max(1, tp + fp),
            'recall': tp / max(1, tp + fn), 'f1': 2*tp / max(1, 2*tp + fp + fn),
            'accuracy': (tp + tn) / max(1, tp + tn + fp + fn)}


def summarize(pairs: list[tuple[np.ndarray, np.ndarray]], operating_conf: float = 0.25) -> dict:
    if not .001 <= operating_conf <= 1:
        raise ValueError("Operating confidence must be in [.001, 1]")
    if not pairs:
        raise ValueError("No evaluated images")
    pairs = [(p[p[:, 4] > operating_conf], t) for p, t in pairs]
    per_class = {}
    for cls, name in enumerate(("LTE", "Radar")):
        rows, counts, operating_tp, operating_fp = [], 0, 0, 0
        frequency_ious = []
        for prediction, target in pairs:
            pred = prediction[prediction[:, 5] == cls]
            truth = target[target[:, 0] == cls]
            correct = match(pred, truth)
            rows.extend(zip(pred[:, 4], correct))
            counts += len(truth)
            selected = pred[:, 4] > operating_conf
            operating_tp += int(correct[selected, 0].sum())
            operating_fp += int(selected.sum() - correct[selected, 0].sum())
            # Auxiliary x-axis IoU for matched operating-point true positives only.
            if len(truth):
                overlaps = box_iou(pred[:, :4], truth[:, 1:])
                used = set()
                for i in np.argsort(-pred[:, 4], kind="stable"):
                    if not selected[i] or not correct[i, 0]:
                        continue
                    candidates = sorted((j for j in range(len(truth)) if j not in used
                                         and overlaps[i, j] >= .5), key=lambda j: -overlaps[i, j])
                    if not candidates:
                        continue
                    j = candidates[0]
                    used.add(j)
                    left, right = max(pred[i, 0], truth[j, 1]), min(pred[i, 2], truth[j, 3])
                    union = (pred[i, 2]-pred[i, 0]) + (truth[j, 3]-truth[j, 1]) - max(0, right-left)
                    frequency_ious.append(float(max(0, right-left) / max(union, 1e-12)))
        scores = np.asarray([r[0] for r in rows])
        correct = np.asarray([r[1] for r in rows], dtype=bool).reshape(-1, 10)
        ap = [average_precision(correct[:, i], scores, counts) for i in range(10)]
        precision = operating_tp / max(1, operating_tp+operating_fp)
        recall = operating_tp/counts if counts else 0.0
        per_class[name] = {"targets": counts, "ap50": ap[0],
                           "ap50_95": float(np.mean(ap)),
                           "precision": precision, "recall": recall,
                           "f1": 2*precision*recall/max(precision+recall, 1e-12) if counts else 0.0,
                           "true_positives": operating_tp, "false_positives": operating_fp,
                           "false_positives_per_image": operating_fp/len(pairs),
                           "matched_frequency_iou": float(np.mean(frequency_ious))
                           if frequency_ious else None}
    valid = list(per_class.values())
    tp = sum(v['true_positives'] for v in valid)
    npred = sum(len(p) for p, _ in pairs)
    ngt = sum(len(t) for _, t in pairs)
    return {"protocol": "cedar-mdi-v1; IoU-greedy; 101-point trapezoidal AP; two-class mean",
            "images": len(pairs), "operating_confidence": operating_conf,
            "target_count": ngt, "prediction_count": npred,
            "precision": tp / npred if npred else 0.0,
            "recall": tp / ngt if ngt else 0.0,
            "map50": float(np.mean([v["ap50"] for v in valid])) if valid else None,
            "map50_95": float(np.mean([v["ap50_95"] for v in valid])) if valid else None,
            "classes": per_class, "overlap": overlap_summary(pairs, operating_conf)}
