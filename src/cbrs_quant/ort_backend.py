"""ORT CUDA, TensorRT+CUDA and CPU routes with explicit placement auditing.

I/O binding keeps model inputs/outputs on the selected device. Provider auditing
runs once before timing; an unavailable or unused requested EP is an error.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import uuid

import numpy as np
import torch

from .io import environment, fingerprint, onnx_precision, read_json, verified_artifact, write_json
from .models import require_cuda

PRIMARY = {"ort_cuda_fp32": "CUDAExecutionProvider", "ort_cuda_fp16": "CUDAExecutionProvider",
           "ort_trt_fp16": "TensorrtExecutionProvider", "ort_trt_int8": "TensorrtExecutionProvider",
           "ort_cpu_fp32": "CPUExecutionProvider"}


def audit_profile(path: Path, primary: str, allow_cpu: bool) -> dict:
    counts = Counter()
    ops = Counter()
    for event in read_json(path):
        args = event.get("args", {})
        provider = args.get("provider")
        if event.get("cat") == "Node" and provider:
            counts[provider] += 1
            ops[f"{provider}:{args.get('op_name', 'unknown')}"] += 1
    if not counts.get(primary):
        raise RuntimeError(f"Requested {primary} executed no profiled nodes; refusing a mislabeled run")
    if not allow_cpu and counts.get("CPUExecutionProvider"):
        raise RuntimeError("CPU fallback was observed but not authorized by --allow-cpu-fallback")
    return {"node_events": dict(counts), "operator_events": dict(ops),
            "profile": str(path), "note": "Untimed placement audit; counts are not FLOP or precision coverage"}


def validate_graph(mode: str, meta: dict) -> None:
    expected = {"ort_cuda_fp32": "fp32", "ort_cuda_fp16": "fp16",
                "ort_trt_fp16": "fp32", "ort_trt_int8": "int8_qdq", "ort_cpu_fp32": "fp32"}
    if onnx_precision(meta) != expected[mode]:
        raise ValueError(f"{mode} requires a {expected[mode]} ONNX graph")


class ORTBackend:
    def __init__(self, path: Path, mode: str, batch: int, cache_dir: Path,
                 allow_cpu_fallback=False, threads=1, workspace_gib=2, optimization_level=3):
        import onnxruntime as ort
        if mode not in PRIMARY or threads < 1 or workspace_gib <= 0:
            raise ValueError("Invalid ORT mode, thread count or workspace")
        if not 0 <= optimization_level <= 5:
            raise ValueError("TensorRT optimization level must be in [0,5]")
        self.meta = verified_artifact(path, "onnx")
        validate_graph(mode, self.meta)
        if self.meta["batch"] != batch:
            raise ValueError("ONNX batch does not match --batch")
        cpu = mode == "ort_cpu_fp32"
        self.device = torch.device("cpu") if cpu else require_cuda()
        self.stream = None if cpu else torch.cuda.current_stream(self.device)
        self.batch, self.mode = batch, mode
        self.dtype = torch.float16 if mode == "ort_cuda_fp16" else torch.float32
        primary = PRIMARY[mode]
        available = ort.get_available_providers()
        required = [primary] + (["CUDAExecutionProvider"] if mode.startswith("ort_trt") else [])
        if any(p not in available for p in required):
            raise RuntimeError(f"Required providers {required}; installed ORT exposes {available}")
        env = environment()
        key = {"onnx_sha256": self.meta["sha256"], "mode": mode, "batch": batch,
               "ort": ort.__version__, "cuda": env.get("cuda_build"),
               "trt": env["packages"].get("tensorrt"), "gpu": env.get("gpu"),
               "cc": env.get("compute_capability"), "machine": env["machine"],
               "cpu_fallback": allow_cpu_fallback, "threads": threads,
               "workspace_gib": workspace_gib, "tf32_override": env["nvidia_tf32_override"],
               "builder_optimization_level": optimization_level}
        disabled_optimizers = "WeightBiasQuantization" if mode == "ort_trt_int8" else ""
        key["disabled_optimizers"] = disabled_optimizers
        if mode.startswith("ort_trt") and not key["trt"]:
            # Native libraries can exist without Python bindings. Their version
            # cannot be verified here, so never reuse their engines across runs.
            key["unknown_trt_session"] = uuid.uuid4().hex
        cache = cache_dir.resolve() / fingerprint(key)
        cache.mkdir(parents=True, exist_ok=True)
        write_json(cache / "identity.json", key)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if disabled_optimizers:
            # ORT 1.22 otherwise inserts INT32 bias DQ nodes unsupported by TRT.
            # Preserve the exported floating bias and original activation/weight QDQ.
            options.add_session_config_entry("optimization.disable_specified_optimizers", disabled_optimizers)
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.enable_profiling = True
        options.profile_file_prefix = str(cache / f"placement-{uuid.uuid4().hex}")
        if not cpu and not allow_cpu_fallback:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers = []
        if mode.startswith("ort_trt"):
            providers.append((primary, {"device_id": 0, "user_compute_stream": str(self.stream.cuda_stream),
                              "trt_fp16_enable": True, "trt_int8_enable": mode.endswith("int8"),
                              "trt_builder_optimization_level": optimization_level,
                              "trt_max_workspace_size": int(workspace_gib*1024**3),
                              "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache),
                              "trt_timing_cache_enable": True, "trt_timing_cache_path": str(cache)}))
        if not cpu:
            providers.append(("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0,
                              "user_compute_stream": str(self.stream.cuda_stream),
                              "do_copy_in_default_stream": True, "cudnn_conv_algo_search": "HEURISTIC"}))
        if cpu or allow_cpu_fallback:
            providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(path.resolve()), sess_options=options, providers=providers)
        self.session.disable_fallback()  # Do not retry an EP failure using a different provider.
        if any(p not in self.session.get_providers() for p in required):
            raise RuntimeError("ORT could not initialize the required EP; inspect CUDA/cuDNN/TensorRT libraries")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one image input and one decoded prediction output")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.shape = (batch, 3, 640, 640)
        type_map = {"tensor(float)": (torch.float32, np.float32),
                    "tensor(float16)": (torch.float16, np.float16)}
        if inputs[0].type not in type_map or type_map[inputs[0].type][0] != self.dtype:
            raise ValueError("ONNX input dtype does not match the selected mode")
        if tuple(inputs[0].shape) != self.shape or outputs[0].type not in type_map:
            raise ValueError("Unsupported ONNX input shape or output dtype")
        out_dtype, out_np = type_map[outputs[0].type]
        out_shape = tuple(self.meta["output_shape"])
        if len(out_shape) != 3 or out_shape[0] != batch or any(n <= 0 for n in out_shape):
            raise ValueError("Invalid output shape in ONNX manifest")
        for actual, expected in zip(outputs[0].shape, out_shape):
            if isinstance(actual, int) and actual != expected:
                raise ValueError("ONNX output shape disagrees with manifest")
        self.output = torch.empty(out_shape, dtype=out_dtype, device=self.device)
        self.binding = self.session.io_binding()
        self.binding.bind_output(self.output_name, self.device.type, 0, out_np,
                                 out_shape, self.output.data_ptr())
        self.input_np = type_map[inputs[0].type][1]
        dummy = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        self(dummy)
        if not cpu:
            self.stream.synchronize()
        if not torch.isfinite(self.output).all():
            raise RuntimeError("ORT produced non-finite output during provider audit")
        profile = Path(self.session.end_profiling())
        audit = audit_profile(profile, primary, cpu or allow_cpu_fallback)
        self.meta.update(mode=mode, batch=batch, route="D" if cpu else ("B" if mode.startswith("ort_trt") else "A"),
                         device=self.device.type, precision=mode.rsplit("_", 1)[1],
                         ort_version=ort.__version__, ort_threads=threads, ort_interop_threads=1,
                         builder_optimization_level=optimization_level if mode.startswith("ort_trt") else None,
                         disabled_optimizers=disabled_optimizers,
                         configured_providers=self.session.get_providers(),
                         provider_options=self.session.get_provider_options(), provider_audit=audit,
                         allow_cpu_fallback=allow_cpu_fallback, cuda_fallback=mode.startswith("ort_trt"),
                         cache_directory=str(cache), tf32=False,
                         precision_note="Q/DQ or FP16 request; provider counts do not establish layer precision",
                         io_binding=True, audit_excluded_from_timing=True)

    @torch.inference_mode()
    def __call__(self, images):
        if (tuple(images.shape) != self.shape or images.dtype != self.dtype
                or images.device != self.device or not images.is_contiguous()):
            raise ValueError("ORT input shape, dtype, device or strides mismatch")
        if self.stream is not None and torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("ORT must execute on the same CUDA stream used to create its session")
        self.binding.bind_input(self.input_name, self.device.type, 0, self.input_np,
                                self.shape, images.data_ptr())
        self.session.run_with_iobinding(self.binding)
        return self.output
