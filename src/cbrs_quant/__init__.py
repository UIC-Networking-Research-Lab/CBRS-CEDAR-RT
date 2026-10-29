"""CBRS GPU inference; optional GPU dependencies are imported on demand."""

import os

# TensorRT EP has no TF32 toggle in ORT 1.22. Set the NVIDIA override before
# importing GPU libraries so ONNX Runtime TensorRT and direct TensorRT use the same policy.
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

__version__ = "0.2.0"
