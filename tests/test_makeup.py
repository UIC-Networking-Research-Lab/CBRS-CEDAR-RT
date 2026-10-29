"""Exercise makeup scheduling using tiny fake workers, never model inference."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbrs_quant.io import digest, fingerprint, read_json, write_json  # noqa: E402
from cbrs_quant.makeup import load_plan, run_makeup, validate_environment, within_run  # noqa: E402


FAKE_WORKER = '''import hashlib, json, sys
from pathlib import Path
job_path = Path(sys.argv[-1])
job = json.loads(job_path.read_text(encoding="utf-8"))
root = job_path.parent
artifacts = {"job.json": hashlib.sha256(job_path.read_bytes()).hexdigest()}
state = {"status": "passed", "phase": "complete", "started_at": "fake-worker-start", "artifact_sha256": artifacts}
if job["kind"] == "export":
    graph = root / "model.fake.onnx"
    graph.write_bytes(b"fake graph; not inference")
    state["graph"] = str(graph)
    artifacts[graph.name] = hashlib.sha256(graph.read_bytes()).hexdigest()
(root / "status.json").write_text(json.dumps(state), encoding="utf-8")
'''


class MakeupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cbrs-makeup-test-")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.run = self.project / "outputs" / "run-fixture"
        self.run.mkdir(parents=True)
        self.source = self.run / "source"
        package = self.source / "cbrs_quant"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "sweep.py").write_text(FAKE_WORKER, encoding="utf-8")
        self.weights = self.run / "checkpoints" / "cbrs_yolo8n.pt"
        self.weights.parent.mkdir()
        self.weights.write_bytes(b"fake checkpoint")
        self.config = {"schema": 1, "output": str(self.run), "source": str(self.source),
                       "source_hashes": {}, "data": str(self.project / "fake_data"),
                       "manifest": str(self.run / "dataset.json"), "manifest_sha256": "fake-dataset-hash",
                       "model_order": ["cbrs_yolo8n"], "batches": [1, 4],
                       "modes": ["torch_cuda_fp32", "ort_cpu_fp32"], "threads": 1, "smoke": True,
                       "minimum_free_gib": 0, "maximum_worker_gib": 8,
                       "weights": {"cbrs_yolo8n": {"sha256": digest(self.weights), "retained_path": str(self.weights)}},
                       "protocol": {"rounds": 1, "scopes": ["model_compute", "detector_pipeline", "image_to_detection"]}}
        self.config["config_sha256"] = fingerprint(self.config)
        write_json(self.run / "config.json", self.config)
        self.path = self.project / "makeup" / "experiments.json"
        self.list = {"schema": 1, "run": "outputs/run-fixture", "experiments": [
            {"model": "cbrs_yolo8n", "batch": 1, "mode": "ort_cpu_fp32", "reason": "failed"},
            {"model": "cbrs_yolo8n", "batch": 1, "mode": "torch_cuda_fp32", "reason": "interrupted"}]}
        write_json(self.path, self.list)

    def stage(self, mode):
        return self.run / "batch_001" / "cbrs_yolo8n" / "results" / mode

    def test_explicit_matrix_validation_and_gpu_priority(self):
        run, _, selected, list_sha = load_plan(self.path, self.project)
        self.assertEqual(list_sha, digest(self.path))
        self.assertEqual(run, self.run)
        self.assertEqual([x["mode"] for x in selected], ["torch_cuda_fp32", "ort_cpu_fp32"])
        for item in ({"model": "../x", "batch": 1, "mode": "ort_cpu_fp32"},
                     {"model": "cbrs_yolo8n", "batch": True, "mode": "ort_cpu_fp32"},
                     {"model": "cbrs_yolo8n", "batch": 8, "mode": "ort_cpu_fp32"},
                     {"model": "cbrs_yolo8n", "batch": 1, "mode": "trt_int8"}):
            invalid = dict(self.list, experiments=[item])
            write_json(self.path, invalid)
            with self.assertRaisesRegex(ValueError, "outside"):
                load_plan(self.path, self.project)
        write_json(self.path, dict(self.list, experiments=self.list["experiments"] * 2))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_plan(self.path, self.project)

    def test_dry_run_leaves_run_unchanged(self):
        before = {p.relative_to(self.run): digest(p) for p in self.run.rglob("*") if p.is_file()}
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_makeup(self.project, self.path, dry_run=True), 0)
        after = {p.relative_to(self.run): digest(p) for p in self.run.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_fake_workers_preserve_old_attempts_and_reuse_completed_makeups(self):
        old = self.stage("torch_cuda_fp32") / "attempt_001"
        old.mkdir(parents=True)
        (old / "worker.log").write_text("original failed work", encoding="utf-8")
        write_json(self.stage("torch_cuda_fp32") / "latest.json", {"status": "failed", "attempt": str(old)})
        before = digest(old / "worker.log")
        with patch("cbrs_quant.makeup.sweep.preflight"), patch("cbrs_quant.makeup.validate_environment"), \
                patch("cbrs_quant.makeup.sweep.summarize_group"), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run_makeup(self.project, self.path, interval=.01), 0)
            first = read_json(self.run / "makeup_progress.json")
            self.assertEqual([item["mode"] for item in first["results"]], ["torch_cuda_fp32", "ort_cpu_fp32"])
            self.assertEqual(run_makeup(self.project, self.path, interval=.01), 0)
        second = read_json(self.run / "makeup_progress.json")
        self.assertTrue(all(item["reused"] for item in second["results"]))
        self.assertEqual(digest(old / "worker.log"), before)
        for mode in ("torch_cuda_fp32", "ort_cpu_fp32"):
            latest = read_json(self.stage(mode) / "latest.json")
            attempt = Path(latest["attempt"])
            marker = read_json(attempt / "makeup.json")
            self.assertTrue(latest["is_makeup"])
            self.assertEqual(latest["makeup_id"], first["makeup_id"])
            self.assertEqual(marker["makeup_id"], first["makeup_id"])
            self.assertTrue(latest["makeup_started_at"] and latest["makeup_finished_at"])
            self.assertEqual(latest["makeup_list_sha256"], digest(self.path))
        self.assertEqual(len(list(self.stage("torch_cuda_fp32").glob("attempt_*"))), 2)
        self.assertEqual(len(list(self.stage("ort_cpu_fp32").glob("attempt_*"))), 1)
        self.assertEqual(len(list((self.run / "batch_001" / "cbrs_yolo8n" / "models" / "export_fp32").glob("attempt_*"))), 1)
        with patch("cbrs_quant.makeup.sweep.preflight"), patch("cbrs_quant.makeup.validate_environment"), \
                patch("cbrs_quant.makeup.sweep.summarize_group"), redirect_stdout(io.StringIO()):
            self.assertEqual(run_makeup(self.project, self.path, interval=.01, force=True), 0)
        self.assertEqual(len(list(self.stage("torch_cuda_fp32").glob("attempt_*"))), 3)
        self.assertEqual(len(list(self.stage("ort_cpu_fp32").glob("attempt_*"))), 2)
        self.assertEqual(len(list((self.run / "batch_001" / "cbrs_yolo8n" / "models" / "export_fp32").glob("attempt_*"))), 1)

    def test_selected_original_success_gets_a_new_attempt(self):
        old = self.stage("torch_cuda_fp32") / "attempt_001"
        write_json(old / "job.json", {"old": "job"})
        write_json(self.stage("torch_cuda_fp32") / "latest.json",
                   {"status": "passed", "attempt": str(old), "artifact_sha256": {"job.json": digest(old / "job.json")}})
        write_json(self.path, dict(self.list, experiments=[self.list["experiments"][1]]))
        with patch("cbrs_quant.makeup.sweep.preflight"), patch("cbrs_quant.makeup.validate_environment"), \
                patch("cbrs_quant.makeup.sweep.summarize_group"), redirect_stdout(io.StringIO()):
            self.assertEqual(run_makeup(self.project, self.path, interval=.01), 0)
        self.assertEqual(Path(read_json(self.stage("torch_cuda_fp32") / "latest.json")["attempt"]).name, "attempt_002")
        self.assertEqual(read_json(old / "job.json"), {"old": "job"})

    def test_failed_export_marks_only_dependent_route_without_using_old_graph(self):
        fake = FAKE_WORKER.replace('if job["kind"] == "export":',
            'if job["kind"] == "export":\n    (root / "status.json").write_text(json.dumps({"status": "failed", "error": "fake export failed"}))\n    raise SystemExit(1)\nif False:')
        (self.source / "cbrs_quant" / "sweep.py").write_text(fake, encoding="utf-8")
        write_json(self.path, dict(self.list, experiments=[self.list["experiments"][0]]))
        with patch("cbrs_quant.makeup.sweep.preflight"), patch("cbrs_quant.makeup.validate_environment"), \
                patch("cbrs_quant.makeup.sweep.summarize_group"), redirect_stdout(io.StringIO()):
            self.assertEqual(run_makeup(self.project, self.path, interval=.01), 1)
        result = read_json(self.stage("ort_cpu_fp32") / "latest.json")
        self.assertEqual(result["status"], "dependency_failed")
        self.assertTrue(result["is_makeup"])
        self.assertFalse(list(self.stage("ort_cpu_fp32").glob("attempt_*")))

    def test_environment_mismatch_rejected_without_inference(self):
        stage = self.stage("torch_cuda_fp32")
        attempt = stage / "attempt_001"
        write_json(attempt / "environment.json", {"gpu": "old GPU"})
        write_json(stage / "latest.json", {"status": "passed", "attempt": str(attempt)})
        fake = type("Result", (), {"returncode": 0, "stdout": json.dumps({"gpu": "new GPU", "cuda_available": True}), "stderr": ""})()
        with patch("subprocess.run", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "Environment differs.*gpu"):
                validate_environment(self.project, self.run, self.config, [self.list["experiments"][1]])

    def test_failed_attempt_can_supply_environment_reference(self):
        stage = self.stage("torch_cuda_fp32")
        attempt = stage / "attempt_001"
        write_json(attempt / "environment.json", {"gpu": "same GPU"})
        write_json(stage / "latest.json", {"status": "failed", "attempt": str(attempt)})
        fake = type("Result", (), {"returncode": 0, "stdout": json.dumps({"gpu": "same GPU", "cuda_available": True}), "stderr": ""})()
        with patch("subprocess.run", return_value=fake):
            validate_environment(self.project, self.run, self.config, [self.list["experiments"][1]])

    def test_paths_outside_original_run_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            within_run(self.run, self.run / ".." / "not-this-run")

    def test_root_launcher_options_and_defaults(self):
        spec = importlib.util.spec_from_file_location("makeup_root_launcher", ROOT / "run_experiments.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        args = launcher.parser().parse_args(["--makeup", "--makeup-list", "custom.json", "--dry-run"])
        self.assertTrue(args.makeup and args.dry_run)
        self.assertEqual(args.makeup_list, "custom.json")
        with patch.object(launcher, "MAKEUP_MODE", True), patch.object(launcher, "FORCE_MAKEUP", True):
            args = launcher.parser().parse_args(["--no-makeup", "--no-force-makeup"])
        self.assertFalse(args.makeup or args.force_makeup)
        with patch("cbrs_quant.makeup.run_makeup") as execute, redirect_stdout(io.StringIO()):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                self.assertEqual(launcher.main(["--makeup", "--smoke", "--models", "cbrs_yolo8n"]), 1)
            self.assertIn("not supported", stderr.getvalue())
            execute.assert_not_called()
        with patch("cbrs_quant.makeup.run_makeup", side_effect=KeyboardInterrupt), \
                patch("cbrs_quant.compact_sweep.compact_makeup", side_effect=KeyboardInterrupt), \
                redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(launcher.main(["--makeup"]), 130)
        self.assertIn("--makeup again", stream.getvalue())
        with patch.object(launcher, "run_sweep") as execute, patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(launcher.main(["--makeup-list", "custom.json"]), 1)
            self.assertIn("require --makeup", stderr.getvalue())
            execute.assert_not_called()
        with patch.object(launcher, "run_sweep") as execute, patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as stopped:
                launcher.main(["--makeup-l", "custom.json"])
            self.assertEqual(stopped.exception.code, 2)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
