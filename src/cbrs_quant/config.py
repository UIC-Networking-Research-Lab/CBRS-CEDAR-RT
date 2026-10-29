"""Dependency-free execution matrix."""

MODES = ("torch_cuda_fp32", "torch_cuda_fp16", "torch_cpu_fp32",
         "ort_cuda_fp32", "ort_cuda_fp16", "ort_trt_fp16", "ort_trt_int8",
         "trt_fp32", "trt_fp16", "trt_int8", "ort_cpu_fp32")

DEFAULT_MODES = ["torch_cuda_fp32", "torch_cuda_fp16", "ort_cuda_fp32", "ort_cuda_fp16",
                 "ort_trt_fp16", "ort_trt_int8", "trt_fp16", "trt_int8", "ort_cpu_fp32"]

DEFAULT_BATCHES = (1, 4, 8, 16, 32)
CPU_MODES = ("torch_cpu_fp32", "ort_cpu_fp32")
MODEL_IDS = tuple(f"cbrs_yolo{family}{size}" for family in (8, 11, 26) for size in "nsx") + tuple(
    f"sig_yolo_v8{size}_f20t20_c100" for size in "nsx")
ROUTE_MODES = {
    "A": ("ort_cuda_fp32", "ort_cuda_fp16"),
    "B": ("ort_trt_fp16", "ort_trt_int8"),
    "C": ("trt_fp16", "trt_int8"),
    "D": ("ort_cpu_fp32",),
}


def execution_phases(modes):
    """All GPU experiments across all models/batches precede CPU experiments."""
    return [(name, selected) for name, selected in (
        ("GPU", [m for m in modes if m not in CPU_MODES]),
        ("CPU", [m for m in modes if m in CPU_MODES]),
    ) if selected]
