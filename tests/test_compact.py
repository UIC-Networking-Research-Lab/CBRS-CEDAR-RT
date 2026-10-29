"""Compact output tests using synthetic reports and fake workers only."""

from contextlib import redirect_stdout
from argparse import Namespace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbrs_quant.compact import (SCOPES, comparisons, initialize_model, put_route, read_csv,
                                save_graph)  # noqa: E402
from cbrs_quant.compact_analysis import plot  # noqa: E402
from cbrs_quant.compact_sweep import compact_makeup, initialize, plan_entries, read_config, run_routes  # noqa: E402
from cbrs_quant.io import digest, read_json, write_json  # noqa: E402
from cbrs_quant.provenance import checkpoint_info, precision_notes  # noqa: E402
import test_analysis  # noqa: E402


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_analysis.AnalysisTests("test_empty_run_is_reportable")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.config.update(data=str(f.root / "data"), manifest=str(f.run / "dataset.json"), threads=1,
                        output=str(f.run), minimum_free_gib=0, maximum_worker_gib=8)
        f.save_config()
        self.output = f.root / "compact"

    def setup_success(self):
        self.fixture.commit(rate=10, quality=.8)
        self.fixture.commit("ort_cuda_fp16", rate=20, quality=.79)

    def populate_compact(self):
        f = self.fixture
        config = dict(f.config, output=str(self.output))
        model = initialize_model(self.output, "cbrs_yolo8n", config, f.dataset, f.weight_hash)
        for mode in ("torch_cuda_fp32", "ort_cuda_fp16"):
            attempt = f.stage(mode) / "attempt_001"
            reports = {name: read_json(attempt / f"{name}.json") for name in ("evaluation", *SCOPES)}
            put_route(model, 1, mode, "passed", reports, {})

    def test_precision_provenance_uses_observed_checkpoint_dtype(self):
        import torch
        for dtype, expected in ((torch.float16, "FP16-stored"), (torch.float32, "FP32-stored")):
            path = self.fixture.root / f"{dtype}.pt"
            torch.save({"ema": {"weight": torch.ones(1, dtype=dtype)}}, path)
            info = checkpoint_info(path)
            self.assertIn(expected, info["onnx_fp32_origin"])
            self.assertIn("INT8", info["comparison_note"])
        mixed = precision_notes({"floating_tensor_dtypes": {"float16": 1, "float32": 1}})
        self.assertIn("see floating_tensor_dtypes", mixed["onnx_fp32_origin"])

    def test_precision_notes_written_for_new_run(self):
        import torch
        checkpoint = self.fixture.run / "checkpoints/cbrs_yolo8n.pt"
        torch.save({"ema": {"weight": torch.ones(1, dtype=torch.float16)}}, checkpoint)
        self.fixture.config["weights"]["cbrs_yolo8n"]["sha256"] = digest(checkpoint)
        self.fixture.save_config()
        assets = self.fixture.root / "assets.json"
        write_json(assets, [{"model_id": "cbrs_yolo8n", "path": str(checkpoint), "sha256": digest(checkpoint)}])
        args = Namespace(output=self.fixture.root / "new", weights_manifest=assets,
                         manifest=self.fixture.run / "dataset.json", project_root=self.fixture.root,
                         batches=[1], modes=["ort_cpu_fp32"], data=self.fixture.root / "data",
                         threads=1, smoke=False, minimum_free_gib=0, maximum_worker_gib=8)
        initialize(args, ["cbrs_yolo8n"])
        folder = args.output / "cbrs_yolo8n"
        meta = read_json(folder / "model.json")
        self.assertEqual(meta["checkpoint_path"], "best.pt")
        self.assertEqual(digest(folder / "best.pt"), digest(checkpoint))
        info = meta["checkpoint_storage"]
        self.assertIn("FP16-stored", info["onnx_fp32_origin"])
        self.assertIn("INT8", info["comparison_note"])

    def test_project_relative_inputs_survive_moving_project(self):
        f = self.fixture
        plan = dict(f.config, output=str(self.output), data="data/yolo_split", manifest="artifacts/dataset.json")
        initialize_model(self.output, "cbrs_yolo8n", plan, f.dataset, f.weight_hash)
        config = read_config(self.output, f.root)
        self.assertEqual(Path(config["data"]), f.root / "data/yolo_split")
        self.assertEqual(Path(config["manifest"]), f.root / "artifacts/dataset.json")

    def test_compact_keeps_all_rounds_and_comparisons(self):
        self.setup_success()
        self.populate_compact()
        model = self.output / "cbrs_yolo8n"
        rows = read_csv(model / "results.csv")
        self.assertEqual(len([r for r in rows if r["status"] == "passed"]), 18)
        self.assertEqual(len(read_csv(model / "comparison.csv")), 6)
        comparison = next(r for r in read_csv(model / "comparison.csv")
                          if r["mode"] == "ort_cuda_fp16" and r["scope"] == "model_compute")
        self.assertEqual(float(comparison["speedup_vs_torch_fp32"]), 2)
        self.assertAlmostEqual(float(comparison["map50_95_delta_pp"]), -1)

    def test_failed_route_retains_partial_measurements_and_is_not_compared(self):
        self.fixture.commit()
        f = self.fixture
        model = initialize_model(self.output, "cbrs_yolo8n", dict(f.config, output=str(self.output)), f.dataset, f.weight_hash)
        attempt = f.stage() / "attempt_001"
        reports = {name: read_json(attempt / f"{name}.json") for name in ("evaluation", *SCOPES)}
        put_route(model, 1, "torch_cuda_fp32", "failed", reports, {"error": "test failure"})
        failed = [r for r in read_csv(model / "results.csv") if r["status"] == "failed"]
        self.assertEqual(len(failed), 9)
        self.assertTrue(all(r["error"] == "test failure" for r in failed))
        self.assertEqual(read_csv(model / "comparison.csv"), [])

    def test_mismatched_timing_signature_has_no_speedup(self):
        self.setup_success()
        self.populate_compact()
        model = self.output / "cbrs_yolo8n"
        rows = read_csv(model / "results.csv")
        for row in rows:
            if row["mode"] == "ort_cuda_fp16":
                row["timing_signature"] = "different"
        result = comparisons(read_json(model / "model.json"), rows)
        row = next(row for row in result if row["mode"] == "ort_cuda_fp16")
        self.assertEqual(row["baseline_status"], "protocol_mismatch")
        self.assertNotIn("speedup_vs_torch_fp32", row)

    def test_interrupted_export_does_not_start_fallback(self):
        f = self.fixture
        config = dict(f.config, output=str(self.output), minimum_free_gib=0)
        model = initialize_model(self.output, "cbrs_yolo8n", config, f.dataset, "weight-hash")
        meta = read_json(model / "model.json")
        meta["checkpoint_path"] = str(f.root / "checkpoint.pt")
        write_json(model / "model.json", meta)
        interrupted = {"state": {"status": "interrupted", "error": "Interrupted"}, "interrupted": True}
        entries = [{"model": "cbrs_yolo8n", "batch": 1, "mode": "ort_cpu_fp32"}]
        with patch("cbrs_quant.compact_sweep.execute", return_value=interrupted) as execute, redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                run_routes(self.output, config, entries, 10)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(read_csv(model / "results.csv")[0]["status"], "interrupted")

    def test_last_route_replaces_previous_rows_in_place(self):
        self.setup_success()
        self.populate_compact()
        model = self.output / "cbrs_yolo8n"
        put_route(model, 1, "ort_cuda_fp16", "failed", {}, {"error": "latest retry failed"})
        rows = [r for r in read_csv(model / "results.csv") if r["batch"] == "1" and r["mode"] == "ort_cuda_fp16"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["error"], "latest retry failed")
        self.assertFalse(any(r["mode"] == "ort_cuda_fp16" for r in read_csv(model / "comparison.csv")))

    def test_gpu_phase_precedes_cpu_for_all_models_and_batches(self):
        config = dict(model_order=["m1", "m2"], batches=[1, 4], modes=["ort_cpu_fp32", "torch_cuda_fp32"])
        modes = [r["mode"] for r in plan_entries(config)]
        self.assertEqual(modes, ["torch_cuda_fp32"] * 4 + ["ort_cpu_fp32"] * 4)

    def test_compact_runner_stores_fake_worker_results_without_attempt_folders(self):
        self.setup_success()
        f = self.fixture
        config = dict(f.config, output=str(self.output), minimum_free_gib=0)
        self.output.mkdir()
        model = initialize_model(self.output, "cbrs_yolo8n", config, f.dataset, "weight-hash")
        checkpoint = model / "best.pt"
        checkpoint.write_bytes(b"fixture")
        meta = read_json(model / "model.json")
        meta["checkpoint_path"] = checkpoint.name
        write_json(model / "model.json", meta)
        base = f.stage("torch_cuda_fp32") / "attempt_001"
        reports = {name: read_json(base / f"{name}.json") for name in ("evaluation", *SCOPES)}
        calls = []

        def fake_execute(task, maximum_worker_gib, interval):
            job = task["job"]
            self.assertEqual(Path(job["weights"]), checkpoint)
            calls.append((job["kind"], job.get("mode")))
            if job["kind"] == "export":
                fake_graph = f.root / "source.onnx"
                fake_graph.write_bytes(b"graph")
                write_json(Path(str(fake_graph)+".json"), {"sha256": digest(fake_graph), "graph_precision": "fp32"})
                target = save_graph(Path(task["model_dir"]), job["batch"], job["precision"], fake_graph)
                return {"state": {"status": "passed", "graph": str(target)}, "reports": {}, "elapsed_seconds": 1}
            payload = json.loads(json.dumps(reports))
            for report in payload.values():
                report["model"]["mode"] = job["mode"]
            return {"state": {"status": "passed"}, "reports": payload, "elapsed_seconds": 1}

        entries = [{"model": "cbrs_yolo8n", "batch": 1, "mode": "torch_cuda_fp32"},
                   {"model": "cbrs_yolo8n", "batch": 1, "mode": "ort_cpu_fp32"}]
        with patch("cbrs_quant.compact_sweep.execute", side_effect=fake_execute), redirect_stdout(io.StringIO()):
            self.assertEqual(run_routes(self.output, config, entries, 10), 0)
        self.assertEqual(calls, [("route", "torch_cuda_fp32"), ("export", None), ("route", "ort_cpu_fp32")])
        self.assertEqual({p.name for p in model.iterdir()},
                         {"results.csv", "model.json", "comparison.csv", "demo", "best.pt", "batch_001_fp32.onnx",
                          "batch_001_fp32.onnx.json"})
        self.assertEqual(len(read_csv(model / "results.csv")), 18)
        self.assertFalse((self.output / "source").exists())

    def test_compact_makeup_reads_only_selected_routes(self):
        self.setup_success()
        self.populate_compact()
        selection = self.fixture.root / "makeup.json"
        write_json(selection, {"schema": 1, "run": str(self.output), "experiments": [
            {"model": "cbrs_yolo8n", "batch": 1, "mode": "ort_cuda_fp16"}]})
        with patch("cbrs_quant.compact_sweep.run_routes") as run, redirect_stdout(io.StringIO()):
            self.assertEqual(compact_makeup(self.fixture.root, selection, dry_run=True), 0)
            run.assert_not_called()
        self.assertFalse((self.output / "source").exists())

    def test_fake_worker_saves_ten_seeded_demos_without_extra_json(self):
        from cbrs_quant import compact_worker, runner, sweep
        model = self.fixture.root / "model"
        (model / "demo").mkdir(parents=True)
        scratch = self.fixture.root / "runtime"
        scratch.mkdir()
        data = self.fixture.root / "pngs"
        data.mkdir()
        records = [{"split": "val", "image": f"image{i}.png"} for i in range(20)]
        manifest = self.fixture.root / "manifest.json"
        write_json(manifest, {"records": records})

        def fake_predict(backend, paths, output, *args, **kwargs):
            output.mkdir()
            for path in paths:
                (output / path.name).write_bytes(b"fake image")
            result = {"records": [{"image": path.name} for path in paths]}
            write_json(output / "predictions.json", result)
            return result

        def fake_run(job_path):
            job = read_json(job_path)
            sweep.stage_state(job_path.parent, phase="demo")
            runner.predict(None, [], job_path.parent / "demo")
            sweep.stage_state(job_path.parent, status="passed", phase="complete")

        class Connection:
            def __init__(self):
                self.messages = []
            def send(self, message):
                self.messages.append(message)
            def close(self):
                pass

        connection = Connection()
        task = {"scratch": str(scratch), "model_dir": str(model), "capture_demo": True,
                "job": {"kind": "route", "batch": 1, "mode": "torch_cuda_fp32",
                        "data": str(data), "manifest": str(manifest)}}
        with patch.object(runner, "predict", side_effect=fake_predict), patch.object(sweep, "_run_worker", side_effect=fake_run):
            compact_worker.run(task, connection)
        result = connection.messages[-1]
        self.assertEqual(result["state"]["status"], "passed")
        self.assertEqual(result["state"]["demo"]["saved_images"], 10)
        self.assertEqual(len(list((model / "demo").glob("*.png"))), 10)
        self.assertFalse((model / "demo/predictions.json").exists())

    @unittest.skipUnless(__import__("importlib.util").util.find_spec("matplotlib"), "matplotlib required")
    def test_analysis_plots_compact_results_without_modifying_them(self):
        self.setup_success()
        self.populate_compact()
        source = self.output / "cbrs_yolo8n/results.csv"
        before = digest(source)
        with redirect_stdout(io.StringIO()):
            files = plot(self.output, self.fixture.root / "figures", ("png", "pdf"), dpi=72, scope="model_compute")
        self.assertEqual(len(files), 2)
        self.assertTrue(all(p.stat().st_size > 100 for p in files))
        self.assertEqual(digest(source), before)


if __name__ == "__main__":
    unittest.main()
