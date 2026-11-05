"""Launcher/scheduler checks using temporary fixtures; no inference or GPU imports."""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("root_launcher", ROOT / "run_experiments.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

from cbrs_quant.config import CPU_MODES, DEFAULT_MODES, execution_phases  # noqa: E402
from cbrs_quant.io import digest, fingerprint, read_json, write_json  # noqa: E402
from cbrs_quant.progress import ConsoleProgress  # noqa: E402
from cbrs_quant.sweep import create_config, job_plan, planned_models, preflight, print_plan, supervise, validate_resume  # noqa: E402


class LauncherTests(unittest.TestCase):
    def test_defaults_cover_all_routes_and_requested_batches(self):
        args = launcher.parser().parse_args([])
        self.assertEqual(args.batches, [1, 4, 8])
        self.assertEqual(args.models, ["all"])
        argv = launcher.sweep_arguments(args)
        modes = argv[argv.index("--modes")+1:argv.index("--threads")]
        self.assertEqual(modes, [m for m in DEFAULT_MODES if m not in CPU_MODES])

    def test_global_defaults_and_cli_overrides(self):
        with patch.object(launcher, "BATCH_SIZES", [8]), patch.object(launcher, "SMOKE", True):
            self.assertEqual(launcher.parser().parse_args([]).batches, [8])
            args = launcher.parser().parse_args(["--batches", "4", "--no-smoke", "--routes", "D"])
        self.assertEqual(args.batches, [4])
        self.assertFalse(args.smoke)
        argv = launcher.sweep_arguments(args)
        modes = argv[argv.index("--modes")+1:argv.index("--threads")]
        self.assertEqual(modes, ["ort_cpu_fp32"])

    def test_exact_modes_override_routes(self):
        args = launcher.parser().parse_args(["--routes", "D", "--modes", "torch_cuda_fp16"])
        argv = launcher.sweep_arguments(args)
        self.assertEqual(argv[argv.index("--modes")+1:argv.index("--threads")], ["torch_cuda_fp16"])
        with patch.object(launcher, "ROUTES", []), patch.object(launcher, "MODES", ["ort_cpu_fp32"]):
            argv = launcher.sweep_arguments(launcher.parser().parse_args([]))
        self.assertEqual(argv[argv.index("--modes")+1:argv.index("--threads")], ["ort_cpu_fp32"])

    def test_dry_run_from_another_cwd_is_read_only_and_dependency_free(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            code = (f"import runpy,sys; sys.argv=['run_experiments.py','--dry-run','--output',{str(output)!r}]; "
                    f"m=runpy.run_path({str(ROOT / 'run_experiments.py')!r}); "
                    "assert m['main']() == 0; "
                    "assert not any(x in sys.modules for x in ('torch','onnxruntime','tensorrt','numpy'))")
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=directory,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("GPU: 288 routes", result.stdout)
            self.assertNotIn("CPU: 60 routes", result.stdout)
            self.assertIn("Total: 288 routes", result.stdout)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(launcher.absolute("artifacts/dataset.json"), ROOT / "artifacts/dataset.json")

    def test_resume_requires_explicit_existing_output(self):
        with self.assertRaisesRegex(ValueError, "Resume requires"):
            launcher.sweep_arguments(launcher.parser().parse_args(["--resume"]))

    def test_missing_inputs_fail_before_creating_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(data=root / "pngs", manifest=root / "dataset.json",
                        modes=["ort_cpu_fp32"], model_order=["cbrs_yolo8n"])
            with patch("importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "Cannot start experiments"):
                    preflight(plan, root / "weights.json")
            self.assertEqual(list(root.iterdir()), [])

    def test_unknown_or_duplicate_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for models in (["unknown"], ["all", "cbrs_yolo8n"], ["cbrs_yolo8n"]*2):
                with self.assertRaises(ValueError):
                    planned_models(SimpleNamespace(models=models, weights_manifest=Path(directory)/"missing.json"))

    def test_cpu_only_plan_does_not_claim_int8_calibration(self):
        plan = dict(model_order=["cbrs_yolo8n"], modes=["ort_cpu_fp32"], batches=[1],
                    threads=1, smoke=False, data="pngs", output="run")
        stream = io.StringIO()
        with redirect_stdout(stream):
            print_plan(plan)
        self.assertIn("calibration: none (no INT8)", stream.getvalue())
        self.assertIn("order: CPU", stream.getvalue())

    def config(self, root):
        weights = root / "checkpoint.pt"
        weights.write_bytes(b"test fixture, not a neural model")
        return dict(output=str(root), data="data", manifest="manifest", manifest_sha256="hash",
                    threads=1, smoke=True, modes=list(DEFAULT_MODES), batches=[1, 4],
                    model_order=["model_a", "model_b"], console=False,
                    weights={name: {"retained_path": str(weights), "sha256": digest(weights)}
                             for name in ("model_a", "model_b")})

    def test_supervisor_runs_every_gpu_group_before_any_cpu_group(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            calls = []

            def fake_execute(config, stage, job, publish):
                calls.append((job["model_id"], job["batch"], job.get("mode"), job["kind"]))
                return {"status": "passed", "graph": str(stage / "fake.onnx")}

            with patch("cbrs_quant.sweep.execute", side_effect=fake_execute), \
                    patch("cbrs_quant.sweep.summarize_group") as summary, redirect_stdout(io.StringIO()):
                result = supervise(config)
            routes = [c for c in calls if c[3] == "route"]
            self.assertEqual(len(routes), 36)
            cpu_indices = [i for i, c in enumerate(routes) if c[2] in CPU_MODES]
            gpu_indices = [i for i, c in enumerate(routes) if c[2] not in CPU_MODES]
            self.assertLess(max(gpu_indices), min(cpu_indices))
            self.assertEqual(result["completed_groups"], 4)
            self.assertEqual(read_json(Path(directory)/"progress.json")["counts"], {"passed": 36})
            for call in summary.call_args_list[:4]:
                self.assertFalse(set(call.args[1]) & set(CPU_MODES))
            for call in summary.call_args_list[4:]:
                self.assertEqual(call.args[1], DEFAULT_MODES)

    def test_phases_reuse_the_same_export_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            graphs = []
            for _, modes in execution_phases(config["modes"]):
                jobs = list(job_plan(config, "model_a", 4, Path("retained.pt"), modes))
                graphs.append(next(job for name, job, _ in jobs if name == "export_fp32"))
            self.assertEqual(graphs[0], graphs[1])
            config["modes"] = ["ort_cpu_fp32"]
            first = next(job_plan(config, "model_a", 4, Path("retained.pt")))[1]
            self.assertEqual(first["export_preference"], "cpu")

    def test_progress_shows_rounds_and_throttles_heartbeat(self):
        progress = ConsoleProgress(interval=10)
        report = dict(planned_routes=540, counts={"passed": 2}, phase="GPU", status="running",
                      active=dict(job="job", attempt="attempt", model="cbrs_yolo8n", batch=4,
                                  mode="ort_cuda_fp16", elapsed_seconds=70,
                                  state={"phase": "model_compute", "detail": {
                                      "round": 1, "rounds": 3, "measurement": "latency"}}))
        stream = io.StringIO()
        with patch("time.monotonic", return_value=100), redirect_stdout(stream):
            progress(report)
            progress(report)
            report["active"]["state"]["detail"]["measurement"] = "throughput"
            progress(report)
        self.assertEqual(len(stream.getvalue().splitlines()), 2)
        self.assertIn("[2/540 routes | GPU]", stream.getvalue())
        self.assertIn("1/3 throughput", stream.getvalue())
        self.assertIn("elapsed 00:01:10", stream.getvalue())

    def test_progress_io_is_outside_measured_loops(self):
        tree = ast.parse((ROOT / "src/cbrs_quant/runner.py").read_text(encoding="utf-8"))
        benchmark = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "benchmark")
        loops = [n for n in ast.walk(benchmark) if isinstance(n, ast.While)]
        self.assertEqual(len(loops), 2)
        for loop in loops:
            self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                 and n.func.id == "progress" for n in ast.walk(loop)))

    def test_saved_model_order_matches_printed_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.pt"
            weights.write_bytes(b"fixture")
            model_order = ["cbrs_yolo8n", "sig_yolo_v8n_f20t20_c100"]
            manifest = root / "weights.json"
            write_json(manifest, [{"model_id": name, "path": weights.name, "sha256": digest(weights)}
                                  for name in model_order])
            data_manifest = root / "data.json"
            write_json(data_manifest, {})
            args = SimpleNamespace(output=root/"run", weights_manifest=manifest,
                                   project_root=root, models=list(reversed(model_order)),
                                   manifest=data_manifest, data=root, batches=[1], modes=["ort_cpu_fp32"],
                                   threads=1, smoke=True, minimum_free_gib=0, maximum_worker_gib=8,
                                   progress_interval=10)
            config = create_config(args)
            self.assertEqual(config["model_order"], args.models)
            self.assertEqual(config["protocol"]["evaluation_split"], "test")

    def test_resume_rejects_mixed_supervisor_and_worker_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, active = root/"snapshot", root/"active"
            snapshot.mkdir()
            active.mkdir()
            for folder in (snapshot, active):
                (folder/"module.py").write_text("value = 1\n")
            config = dict(output=str(root), source=str(snapshot),
                          protocol={"evaluation_split": "test", "quality_protocol": "cedar-mdi-v1"},
                          source_hashes={"module.py": digest(snapshot/"module.py")})
            config["config_sha256"] = fingerprint(config)
            validate_resume(config, root, active)
            (active/"module.py").write_text("value = 2\n")
            with self.assertRaisesRegex(ValueError, "Current code differs"):
                validate_resume(config, root, active)
            with self.assertRaisesRegex(ValueError, "Run was moved"):
                validate_resume(config, root/"elsewhere", active)


if __name__ == "__main__":
    unittest.main()
