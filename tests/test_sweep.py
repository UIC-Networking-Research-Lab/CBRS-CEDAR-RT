"""Sweep dependency ordering and crash-safe restart behavior."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cbrs_quant.config import DEFAULT_MODES
from cbrs_quant.io import digest, write_json
from cbrs_quant.sweep import execute, exclusive_lock, job_plan, valid_previous


class SweepTests(unittest.TestCase):
    def test_dependencies_precede_routes(self):
        config = dict(output="output", data="data", manifest="manifest", manifest_sha256="hash",
                      threads=1, smoke=False, modes=DEFAULT_MODES,
                      weights={"model": {"sha256": "weights_hash"}})
        completed, routes = set(), []
        for name, job, dependency in job_plan(config, "model", 64, Path("model.pt")):
            if dependency:
                self.assertIn(dependency, completed)
            self.assertEqual(job["batch"], 64)
            self.assertEqual(job["weights_sha256"], "weights_hash")
            completed.add(name)
            if job["kind"] == "route":
                routes.append(name)
        self.assertCountEqual(routes, DEFAULT_MODES)
        self.assertEqual(len(completed), len(DEFAULT_MODES)+3)

    def test_completed_and_failed_routes_not_repeated_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            artifact = stage / "test.json"
            write_json(artifact, {"value": 123})
            write_json(stage / "job.json", {})
            for status in ("passed", "failed", "dependency_failed"):
                write_json(stage / "latest.json", {"status": status, "attempt": str(stage),
                           "artifact_sha256": {"test.json": digest(artifact)}})
                with patch("subprocess.Popen") as process:
                    self.assertEqual(execute({}, stage, {}, lambda **kw: None)["status"], status)
                    process.assert_not_called()

    def test_modified_or_missing_passed_artifacts_require_new_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.onnx"
            artifact.write_bytes(b"model")
            report = {"attempt": directory, "artifact_sha256": {artifact.name: digest(artifact)}}
            self.assertTrue(valid_previous(report))
            artifact.write_bytes(b"tampered")
            self.assertFalse(valid_previous(report))
            artifact.unlink()
            self.assertFalse(valid_previous(report))

    def test_rebuilt_export_invalidates_dependent_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_path = root / "job.json"
            job = {"graph": "old/model.onnx", "mode": "ort_cpu_fp32"}
            write_json(job_path, dict(job, export_device="cuda"))
            report = {"attempt": directory, "artifact_sha256": {job_path.name: digest(job_path)}}
            self.assertTrue(valid_previous(report, job))
            self.assertFalse(valid_previous(report, dict(job, graph="new/model.onnx")))

    def test_exclusive_supervisor_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock"
            with exclusive_lock(path):
                with self.assertRaises(OSError):
                    with exclusive_lock(path):
                        pass
            with exclusive_lock(path):
                pass

    def test_one_wrong_image_cannot_hide_in_large_batch(self):
        import numpy as np
        import torch
        from cbrs_quant.smoke import smoke

        class Backend:
            device, dtype = torch.device("cpu"), torch.float32
            batch = 64
            meta = {"model_id": "test", "weights_sha256": "hash", "layout": "xyxy_scores"}

            def __call__(self, images):
                output = torch.ones(self.batch, 6, 4)
                if self.batch > 1:
                    output[0] *= 1.3
                return output

        reference = Backend()
        reference.batch = 1
        with patch("cbrs_quant.smoke.TorchBackend", return_value=reference), \
                patch("cbrs_quant.smoke.digest", return_value="hash"), \
                patch("cbrs_quant.smoke.load_png", return_value=np.zeros((1, 1, 3))), \
                patch("cbrs_quant.smoke.to_tensor", return_value=None), \
                patch("cbrs_quant.smoke.detections", return_value=[]):
            with self.assertRaisesRegex(ValueError, "Raw output parity failed"):
                smoke(Backend(), [Path("test.png")], Path("test.pt"),
                      reference_device="cpu", reference_batch=1)


if __name__ == "__main__":
    unittest.main()
