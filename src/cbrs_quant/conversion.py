"""Optional offline IQ/SigMF -> PNG reproduction. Never part of timed inference.

WARNING: parameters, magnitude-to-dB formula, orientation, interpolation and
library versions affect pixels and may invalidate pretrained models. A changed
renderer requires a new profile, new PNGs and renewed accuracy validation.
This deliberately retains the original renderer's implicit SciPy/Matplotlib defaults.
"""

from __future__ import annotations

import importlib.metadata
import io
from pathlib import Path

import numpy as np

from .data import load_png
from .io import digest, environment, fingerprint, read_json, reserve_output, sidecar, write_json


def render_iq(iq: np.ndarray, profile: dict) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal

    if (iq.ndim != 1 or not np.iscomplexobj(iq) or len(iq) < profile["nperseg"]
            or not np.isfinite(iq).all()):
        raise ValueError("Expected a finite, complex 1-D IQ array with at least nperseg samples")
    _, _, spectrum = signal.spectrogram(
        iq, fs=profile["sample_rate_hz"], nperseg=profile["nperseg"],
        noverlap=profile["noverlap"], return_onesided=False, mode="magnitude")
    spectrum = 10 * np.log10(np.fft.fftshift(spectrum, axes=0) + 1e-10)
    low, high = np.percentile(spectrum, profile["percentiles"])
    width, height = profile["image_size"]
    buffer = io.BytesIO()
    # rc_context prevents application-level plotting styles from changing the pixels.
    with plt.rc_context(matplotlib.rcParamsDefault):
        figure = plt.figure(figsize=(width/profile["dpi"], height/profile["dpi"]),
                             dpi=profile["dpi"])
        try:
            axes = figure.add_axes([0, 0, 1, 1])
            axes.imshow(spectrum.T, aspect="auto", origin="upper", cmap="viridis",
                        vmin=low, vmax=high)
            axes.axis("off")
            axes.set_xticks([])
            axes.set_yticks([])
            plt.savefig(buffer, format="png", dpi=profile["dpi"], bbox_inches="tight",
                        pad_inches=0, facecolor="black")
        finally:
            plt.close(figure)
    return buffer.getvalue()


def convert(source: Path, output: Path, profile_path: Path, metadata: Path | None = None,
            allow_version_mismatch: bool = False, cbrs_rate_override: bool = False) -> dict:
    profile = read_json(profile_path)
    fixed = {"image_size": [640, 640], "mode": "magnitude", "transpose": True,
             "origin": "upper", "cmap": "viridis", "db": "10*log10(magnitude+1e-10)"}
    if any(profile.get(k) != v for k, v in fixed.items()):
        raise ValueError("Unsupported renderer profile; create and validate a new implementation")
    mismatches = {name: {"expected": version, "actual": importlib.metadata.version(name)}
                  for name, version in profile["versions"].items()
                  if importlib.metadata.version(name) != version}
    if mismatches and not allow_version_mismatch:
        raise ValueError(f"Renderer version mismatch: {mismatches}; use the conversion environment")
    if metadata:
        info = read_json(metadata)["global"]
        if info.get("core:datatype") != "cf32_le":
            raise ValueError("Only SigMF cf32_le is supported")
        rate = info.get("core:sample_rate")
        if rate != profile["sample_rate_hz"] and not cbrs_rate_override:
            raise ValueError("SigMF sample rate differs; --cbrs-rate-override is only for the known CBRS metadata error")
    if source.suffix == ".npy":
        iq = np.load(source, allow_pickle=False)
    else:
        if source.stat().st_size % 8:
            raise ValueError("cf32_le input byte length must be divisible by eight")
        iq = np.fromfile(source, dtype="<c8")
    reserve_output(output)
    output.write_bytes(render_iq(iq, profile))
    load_png(output)
    report = {"kind": "converted_png", "sha256": digest(output),
              "source_sha256": digest(source), "metadata_sha256": digest(metadata) if metadata else None,
              "profile_sha256": fingerprint(profile), "profile": profile,
              "version_mismatches": mismatches, "cbrs_rate_override": cbrs_rate_override,
              "environment": environment(), "labels": "not generated; use archived dataset generator"}
    write_json(sidecar(output), report)
    return report


def verify_png(actual: Path, expected: Path, tolerance: int = 0) -> dict:
    if not 0 <= tolerance <= 255:
        raise ValueError("Pixel tolerance must be in [0,255]")
    difference = np.abs(load_png(actual).astype(np.int16)-load_png(expected).astype(np.int16))
    return {"passed": bool(difference.max() <= tolerance), "tolerance": tolerance,
            "max_absolute_error": int(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "different_channels": int(np.count_nonzero(difference))}
