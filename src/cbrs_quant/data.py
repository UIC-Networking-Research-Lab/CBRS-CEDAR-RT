"""PNG contract and portable dataset manifests.

WARNING: pretrained weights depend on the original spectral/rendering rules and
library versions. Equal image dimensions do not imply compatible model inputs.
Never resize, recolor, or regenerate inputs silently.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from .io import digest, fingerprint, read_json, write_json


def load_png(path: str | Path, size: tuple[int, int] = (640, 640)) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != size:
            raise ValueError(f"Expected {size} PNG, got {image.format}/{image.size}: {path}")
        # Original matplotlib PNGs have an opaque alpha channel.
        if image.mode == "RGBA" and image.getextrema()[3] != (255, 255):
            raise ValueError(f"Non-opaque PNG is not a training-compatible input: {path}")
        if "transparency" in image.info:
            raise ValueError(f"PNG transparency metadata is not supported: {path}")
        if image.mode not in ("RGB", "RGBA"):
            raise ValueError(f"Expected RGB or opaque RGBA, got {image.mode}: {path}")
        return np.array(image.convert("RGB"), dtype=np.uint8)


def load_labels(path: str | Path, size: tuple[int, int] = (640, 640)) -> np.ndarray:
    """Return class, x1, y1, x2, y2; missing labels are errors, empty files are valid."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = np.array([float(x) for x in line.split()], dtype=np.float64)
        if (len(values) != 5 or not np.isfinite(values).all() or values[0] not in (0, 1)
                or np.any(values[1:] < 0) or np.any(values[1:] > 1)
                or np.any(values[3:] <= 0)):
            raise ValueError(f"Invalid YOLO label in {path}: {line}")
        cls, x, y, w, h = values
        corners = np.array([x-w/2, y-h/2, x+w/2, y+h/2])
        if np.any(corners < -1e-5) or np.any(corners > 1+1e-5):
            raise ValueError(f"Label extends outside the image: {path}")
        corners = corners.clip(0, 1) * np.array([size[0], size[1], size[0], size[1]])
        rows.append([cls, *corners])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def make_manifest(root: Path, profile_path: Path, output: Path) -> dict:
    """Hash every PNG and label; explicitly supplied profile is a provenance assertion."""
    profile = read_json(profile_path)
    if profile.get("image_size") != [640, 640] or profile.get("classes") != ["LTE", "Radar"]:
        raise ValueError("This release requires 640x640 and classes [LTE, Radar]")
    records, stems, image_hashes = [], set(), {}
    for split in ("train", "val", "test"):
        paths = sorted((root / split / "images").glob("*.png"))
        if not paths:
            raise ValueError(f"No PNG files in {root / split / 'images'}")
        for path in paths:
            label = root / split / "labels" / (path.stem + ".txt")
            load_png(path)
            labels = load_labels(label)
            if path.stem in stems:
                raise ValueError(f"Sample name occurs in multiple splits: {path.stem}")
            stems.add(path.stem)
            sha = digest(path)
            # Detect leakage without rejecting repeats within the same split.
            if sha in image_hashes and image_hashes[sha] != split:
                raise ValueError(f"Identical PNG bytes occur across splits: {path}")
            image_hashes[sha] = split
            records.append({"split": split, "image": path.relative_to(root).as_posix(),
                            "label": label.relative_to(root).as_posix(), "image_sha256": sha,
                            "label_sha256": digest(label), "objects": len(labels)})
    manifest = {"schema": 1, "profile": profile, "profile_sha256": fingerprint(profile),
                "records": records, "records_sha256": fingerprint(records),
                "provenance_status": "user_asserted_profile; pixel equivalence not inferred"}
    write_json(output, manifest)
    return manifest


class Dataset:
    def __init__(self, root: Path, manifest: Path, split: str, verify: bool = True):
        self.root = root.resolve()
        self.manifest = read_json(manifest)
        if (fingerprint(self.manifest["records"]) != self.manifest["records_sha256"]
                or fingerprint(self.manifest["profile"]) != self.manifest["profile_sha256"]):
            raise ValueError("Dataset manifest has been modified")
        self.records = [r for r in self.manifest["records"] if r["split"] == split]
        if not self.records:
            raise ValueError(f"Empty split: {split}")
        for record in self.records:
            for key in ("image", "label"):
                path = self.path(record[key])
                if verify and digest(path) != record[key + "_sha256"]:
                    raise ValueError(f"Dataset changed since validation: {path}")

    def path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Dataset path escapes root: {relative}")
        return path

    def calibration(self, count: int, seed: int) -> list[dict]:
        if any(r["split"] != "train" for r in self.records):
            raise ValueError("Calibration must use the train split")
        if not 1 <= count <= len(self.records):
            raise ValueError(f"Calibration count must be in [1, {len(self.records)}]")
        return random.Random(seed).sample(self.records, count)


def to_tensor(images: list[np.ndarray], device, dtype):
    import torch
    array = np.stack(images).transpose(0, 3, 1, 2).copy()
    return torch.from_numpy(array).to(device=device, dtype=dtype).div_(255)
