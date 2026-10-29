"""Provider/precision/artifact contracts; real ORT execution is tested separately."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cbrs_quant.archive import archive_models
from cbrs_quant.io import digest, read_json, sidecar, write_json
from cbrs_quant.ort_backend import audit_profile, validate_graph
from cbrs_quant.runtime import synchronize


class ORTContracts(unittest.TestCase):
    def test_cli_help_does_not_import_torch(self):
        import subprocess
        import sys
        launcher = Path(__file__).resolve().parents[1] / "cli.py"
        code = (f"import sys, runpy; runpy.run_path({str(launcher)!r}); "
                "from cbrs_quant.cli import parser; parser(); assert 'torch' not in sys.modules")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wheel_distribution_validation(self):
        import importlib.util
        import zipfile
        script = Path(__file__).resolve().parents[1] / "scripts/setup_jetson.py"
        spec = importlib.util.spec_from_file_location("setup_jetson", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            wheel = Path(folder) / "misleading-gpu.whl"
            with zipfile.ZipFile(wheel, "w") as stream:
                stream.writestr("onnxruntime-1.22.dist-info/METADATA", "Name: onnxruntime\nVersion: 1.22\n")
            self.assertEqual(module.wheel_distribution(wheel), "onnxruntime")

    def test_missing_or_wrong_provider_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.json"
            write_json(path, [{"cat": "Node", "args": {"provider": "CPUExecutionProvider", "op_name": "Conv"}}])
            with self.assertRaises(RuntimeError):
                audit_profile(path, "CUDAExecutionProvider", True)
            with self.assertRaises(RuntimeError):
                audit_profile(path, "CPUExecutionProvider", False)
            self.assertEqual(audit_profile(path, "CPUExecutionProvider", True)["node_events"],
                             {"CPUExecutionProvider": 1})

    def test_precision_mismatch_rejected(self):
        validate_graph("ort_cpu_fp32", {"graph_precision": "fp32"})
        with self.assertRaises(ValueError):
            validate_graph("ort_cpu_fp32", {"graph_precision": "int8_qdq"})
        with self.assertRaises(ValueError):
            validate_graph("ort_cuda_fp16", {"graph_precision": "fp32"})

    def test_cpu_synchronization_never_touches_cuda(self):
        from types import SimpleNamespace
        with patch("torch.cuda.synchronize", side_effect=AssertionError("CPU touched CUDA")):
            synchronize(SimpleNamespace(device=torch.device("cpu")))

    def test_archive_retains_matching_pt_and_onnx(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pt, onnx = root / "original.pt", root / "model.onnx"
            pt.write_bytes(b"checkpoint fixture")
            onnx.write_bytes(b"graph fixture")
            meta = {"kind": "onnx", "sha256": digest(onnx), "model_id": "example",
                    "weights_sha256": digest(pt)}
            write_json(sidecar(onnx), meta)
            result = archive_models(pt, [onnx], [], root / "retained")
            self.assertEqual(digest(root / "retained/checkpoint.pt"), digest(pt))
            self.assertEqual(digest(root / "retained/model.onnx"), digest(onnx))
            self.assertEqual(len(result["files"]), 3)
            self.assertEqual(read_json(root / "retained/manifest.json")["model_id"], "example")
            pt.write_bytes(b"different checkpoint")
            with self.assertRaises(ValueError):
                archive_models(pt, [onnx], [], root / "bad")

    def test_export_head_matches_eager_cedar(self):
        from cbrs_quant.models import CEDARExportHead
        from cbrs_quant.vendor.asym_yolo import AsymDetect
        torch.manual_seed(7)
        head = AsymDetect(nc=2, ch=16, stride_freq=32, stride_time=32,
                          reg_max_freq=4, reg_max_time=4).eval()
        value = torch.randn(1, 16, 4, 4)
        with torch.no_grad():
            eager = head(value)[0]
            exported = CEDARExportHead(head)(value)
        torch.testing.assert_close(exported, eager, rtol=0, atol=0)
        self.assertEqual(exported.dtype, torch.float32)

    def test_cpu_benchmark_no_cuda_and_correct_completion_count(self):
        from types import SimpleNamespace
        from PIL import Image
        from cbrs_quant.runner import benchmark
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.png"
            Image.new("RGB", (640, 640), (1, 2, 3)).save(path)
            dataset = SimpleNamespace(records=[{"image": "input.png", "split": "val"}],
                                      path=lambda _: path, manifest={"records_sha256": "x", "profile_sha256": "p"})
            class Backend:
                batch = 1
                device = torch.device("cpu")
                dtype = torch.float32
                meta = {"layout": "end2end", "device": "cpu"}
                def __call__(self, value):
                    return torch.zeros((1, 1, 6))
            with patch("torch.cuda.synchronize", side_effect=AssertionError("CPU touched CUDA")):
                result = benchmark(Backend(), dataset, "model_compute", warmup=1, min_images=2,
                                   seconds=.001, rounds=1, pool_size=1)
            self.assertGreaterEqual(result["rounds"][0]["throughput_images"], 2)
            self.assertEqual(result["latency_clock"], "synchronized_wall")
            self.assertIsNone(result["torch_peak_allocated_bytes"])


if __name__ == "__main__":
    unittest.main()
