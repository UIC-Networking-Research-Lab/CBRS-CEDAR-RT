"""Single-stream GPU backends with a common tensor interface."""

from __future__ import annotations

from pathlib import Path

import torch

from .io import environment, verified_artifact
from .models import load_model, require_cuda
from .config import MODES


def is_cpu_mode(mode: str) -> bool:
    return mode in ("ort_cpu_fp32", "torch_cpu_fp32")


def synchronize(backend):
    if backend.device.type == "cuda":
        torch.cuda.synchronize(backend.device)


class TorchBackend:
    def __init__(self, model_id: str, weights: Path, mode: str, batch: int):
        self.device = torch.device("cpu") if is_cpu_mode(mode) else require_cuda()
        self.dtype = torch.float16 if mode == "torch_cuda_fp16" else torch.float32
        self.model, self.meta = load_model(model_id, weights)
        self.model.to(device=self.device, dtype=self.dtype)
        self.meta.update(mode=mode, batch=batch, tf32=False, route="reference",
                         precision="fp16" if self.dtype == torch.float16 else "fp32",
                         device=self.device.type)
        self.batch = batch

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape != (self.batch, 3, 640, 640):
            raise ValueError("Input batch/shape does not match backend")
        return self.model(images)


class TensorRTBackend:
    """TensorRT 10.x named I/O. Returned output storage is reused on the next call."""

    def __init__(self, engine_path: Path, mode: str, batch: int):
        import tensorrt as trt
        if int(trt.__version__.split(".")[0]) != 10:
            raise RuntimeError("This release supports TensorRT 10.x; validate other majors first")
        self.device = require_cuda()
        self.meta = verified_artifact(engine_path, "engine")
        self.meta.update(route="C", device="cuda", precision=mode.removeprefix("trt_"))
        if self.meta["mode"] != mode or self.meta["batch"] != batch:
            raise ValueError("Engine precision/batch does not match the requested run")
        env = environment()
        built = self.meta["environment"]
        if (built["gpu"] != env["gpu"] or built["packages"]["tensorrt"] != trt.__version__
                or built["compute_capability"] != env["compute_capability"]):
            raise ValueError("Rebuild the engine on this GPU and TensorRT version")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT could not deserialize the engine")
        self.context = self.engine.create_execution_context()
        self.batch = batch
        types = {trt.float32: torch.float32, trt.float16: torch.float16,
                 trt.int32: torch.int32, trt.int64: torch.int64}
        inputs, outputs = [], []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(n <= 0 for n in shape):
                raise ValueError("Only static-shape engines are supported")
            dtype = types[self.engine.get_tensor_dtype(name)]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append((name, shape, dtype))
            else:
                outputs.append((name, torch.empty(shape, device=self.device, dtype=dtype)))
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one image input and one decoded prediction output")
        self.input_name, self.input_shape, self.dtype = inputs[0]
        self.output_name, self.output = outputs[0]
        if self.input_shape != (batch, 3, 640, 640):
            raise ValueError("Engine input shape does not match the detector contract")
        if not self.context.set_tensor_address(self.output_name, self.output.data_ptr()):
            raise RuntimeError("Could not bind TensorRT output")

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if (tuple(images.shape) != self.input_shape or images.dtype != self.dtype
                or images.device != self.device or not images.is_contiguous()):
            raise ValueError("Incorrect TensorRT input shape, dtype, device or strides")
        if not self.context.set_tensor_address(self.input_name, images.data_ptr()):
            raise RuntimeError("Could not bind TensorRT input")
        if not self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return self.output


def create_backend(mode: str, model_id: str, weights: Path | None,
                   engine: Path | None, batch: int = 1, onnx: Path | None = None,
                   cache_dir: Path = Path("artifacts/ort_cache"),
                   allow_cpu_fallback: bool = False, threads: int = 1, workspace_gib: float = 2,
                   optimization_level: int = 3):
    if batch < 1 or mode not in MODES:
        raise ValueError("Invalid mode or batch size")
    if mode.startswith("torch"):
        if weights is None:
            raise ValueError("PyTorch mode requires --weights")
        return TorchBackend(model_id, weights, mode, batch)
    if mode.startswith("ort_"):
        from .ort_backend import ORTBackend
        if onnx is None:
            raise ValueError("ONNX Runtime requires --onnx")
        backend = ORTBackend(onnx, mode, batch, cache_dir, allow_cpu_fallback, threads, workspace_gib, optimization_level)
        if backend.meta["model_id"] != model_id:
            raise ValueError("ONNX model identifier mismatch")
        return backend
    if engine is None:
        raise ValueError("TensorRT mode requires --engine")
    backend = TensorRTBackend(engine, mode, batch)
    if backend.meta["model_id"] != model_id:
        raise ValueError("Engine model identifier mismatch")
    return backend
