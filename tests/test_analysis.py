"""Synthetic report fixtures; never start inference or read production model assets."""

from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbrs_quant.analysis import (CLASS_METRICS, SCOPES, collect, open_snapshot, select_run, snapshot_directory,
                                 tables, write_report)  # noqa: E402
from cbrs_quant.io import digest, fingerprint, read_json, write_json  # noqa: E402


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cbrs-analysis-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "outputs" / "run-original"
        self.run.mkdir(parents=True)
        checkpoint = self.run / "checkpoints/cbrs_yolo8n.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"synthetic checkpoint")
        self.weight_hash = digest(checkpoint)
        self.dataset = {"records": [{"split": "val"}], "profile": {"id": "fixture"}}
        self.dataset.update(records_sha256=fingerprint(self.dataset["records"]),
                            profile_sha256=fingerprint(self.dataset["profile"]))
        write_json(self.run / "dataset.json", self.dataset)
        self.config = dict(schema=1, created_at=datetime.now(timezone.utc).isoformat(),
                           model_order=["cbrs_yolo8n"], batches=[1, 4],
                           modes=["torch_cuda_fp32", "ort_cuda_fp16", "ort_cpu_fp32"], smoke=False,
                           protocol={"evaluation_split": "val", "scopes": list(SCOPES), "rounds": 3, "seconds": 30},
                           weights={"cbrs_yolo8n": {"sha256": self.weight_hash}},
                           manifest_sha256=digest(self.run / "dataset.json"))
        self.save_config()
        write_json(self.run / "progress.json", {"status": "running"})

    def save_config(self):
        self.config.pop("config_sha256", None)
        self.config["config_sha256"] = fingerprint(self.config)
        write_json(self.run / "config.json", self.config)

    def stage(self, mode="torch_cuda_fp32", batch=1):
        return self.run / f"batch_{batch:03d}" / "cbrs_yolo8n" / "results" / mode

    def commit(self, mode="torch_cuda_fp32", batch=1, rate=10, quality=.8, change=None):
        stage = self.stage(mode, batch)
        attempt = stage / "attempt_001"
        common = dict(dataset_sha256=self.dataset["records_sha256"], profile_sha256=self.dataset["profile_sha256"],
                      split="val", smoke_only_subset=False, environment={"gpu": "fixture GPU"},
                      model={"model_id": "cbrs_yolo8n", "mode": mode, "batch": batch, "weights_sha256": self.weight_hash,
                             "precision": "fp16" if mode.endswith("fp16") else "fp32", "device": "cuda", "route": "reference"},
                      nms_iou=.45, max_det=300)
        metrics = dict(images=1, map50=quality, map50_95=quality, protocol="fixture AP", operating_confidence=.25,
                       classes={name: dict.fromkeys(CLASS_METRICS, quality) for name in ("LTE", "Radar")})
        reports = {"evaluation": dict(deepcopy(common), kind="evaluation", evaluated_images=1,
                                       metrics=metrics, collection_confidence=.001)}
        for scope in SCOPES:
            trials = [dict(images_per_second=rate, throughput_images=1200, throughput_seconds=1200/rate,
                           latency_requests=1200, latency_ms_mean=1000/rate,
                           latency_ms_p50=1000/rate, latency_ms_p95=1200/rate, latency_ms_p99=1500/rate)] * 3
            reports[scope] = dict(deepcopy(common), kind="benchmark", scope=scope, rounds=deepcopy(trials),
                                  protocol_version=2, latency_clock="synchronized_wall", warmup_requests=50,
                                  minimum_images=1000, minimum_seconds=30, pool_images=64, pool_repeats=True, confidence=.25)
        if change:
            change(reports)
        for name, report in reports.items():
            write_json(attempt / f"{name}.json", report)
        state = dict(status="passed", phase="complete", attempt=str(attempt),
                     artifact_sha256={f"{name}.json": digest(attempt / f"{name}.json") for name in reports})
        write_json(attempt / "status.json", state)
        write_json(stage / "latest.json", state)
        return attempt

    def scan(self):
        return collect(self.run, progress=lambda *args, **kwargs: None)

    def test_empty_run_is_reportable(self):
        snapshot = self.scan()
        self.assertEqual(len(snapshot["routes"]), 6)
        self.assertEqual(snapshot["accepted"], [])
        output = snapshot_directory(self.run, self.root / "analysis", self.root / "outputs")
        summary = write_report(snapshot, output, "torch_cuda_fp32", *tables(snapshot, "torch_cuda_fp32"), [])
        self.assertEqual(summary["counts"], {"pending": 6})
        self.assertIn("images_per_second_mean", (output / "performance.csv").read_text(encoding="utf-8-sig"))
        self.assertTrue((output / "index.html").exists())

    def test_numeric_results_and_baselines_are_batch_specific(self):
        self.commit(rate=10, quality=.8)
        self.commit("ort_cuda_fp16", rate=20, quality=.79)
        self.commit("ort_cuda_fp16", batch=4, rate=20, quality=.79)
        quality, timing, trials = tables(self.scan(), "torch_cuda_fp32")
        current = next(r for r in timing if r["mode"] == "ort_cuda_fp16" and r["batch"] == 1)
        self.assertEqual(current["speedup_vs_baseline"], 2)
        self.assertAlmostEqual(current["map50_95_delta_pp"], -1)
        self.assertEqual(current["p95_ms_mean"], 60)
        self.assertEqual(current["images_per_second_std"], 0)
        missing = next(r for r in timing if r["batch"] == 4)
        self.assertEqual(missing["baseline_status"], "missing")
        self.assertNotIn("speedup_vs_baseline", missing)
        self.assertEqual((len(quality), len(timing), len(trials)), (3, 9, 27))

    def test_makeup_provenance_is_available_in_all_result_tables(self):
        self.commit()
        latest_path = self.stage() / "latest.json"
        latest = read_json(latest_path)
        latest.update(is_makeup=True, makeup_started_at="start-utc", makeup_finished_at="end-utc", makeup_id="retry-1")
        write_json(latest_path, latest)
        snapshot = self.scan()
        quality, timing, rounds = tables(snapshot, "torch_cuda_fp32")
        for row in (snapshot["routes"][0], quality[0], timing[0], rounds[0]):
            self.assertEqual(row["execution_kind"], "makeup")
            self.assertEqual(row["execution_started_at"], "start-utc")
            self.assertEqual(row["makeup_finished_at"], "end-utc")
            self.assertEqual(row["makeup_id"], "retry-1")

    def test_uncommitted_retry_supersedes_previous_success(self):
        self.commit()
        newer = self.stage() / "attempt_002"
        write_json(newer / "status.json", {"status": "running", "phase": "full_validation"})
        snapshot = self.scan()
        self.assertEqual(snapshot["accepted"], [])
        self.assertEqual(snapshot["routes"][0]["status"], "incomplete")
        self.assertTrue(snapshot["routes"][0]["attempt"].endswith("attempt_002"))

    def test_failed_retry_does_not_fall_back_to_success(self):
        self.commit()
        newer = self.stage() / "attempt_002"
        newer.mkdir()
        write_json(self.stage() / "latest.json", {"status": "failed", "attempt": str(newer), "error": "fixture OOM"})
        snapshot = self.scan()
        self.assertFalse(snapshot["accepted"])
        self.assertEqual(snapshot["routes"][0]["status"], "failed")

    def test_corrupt_or_missing_success_reports_are_excluded(self):
        attempt = self.commit()
        (attempt / "model_compute.json").write_text("{}", encoding="utf-8")
        snapshot = self.scan()
        self.assertFalse(snapshot["accepted"])
        self.assertIn("hash mismatch", snapshot["routes"][0]["error"])
        (attempt / "model_compute.json").unlink()
        self.assertEqual(self.scan()["routes"][0]["status"], "invalid")

    def test_windows_saved_path_can_be_read_on_another_machine(self):
        self.commit()
        latest = read_json(self.stage() / "latest.json")
        latest["attempt"] = r"Z:\old_machine\results\torch_cuda_fp32\attempt_001"
        write_json(self.stage() / "latest.json", latest)
        self.assertEqual(len(self.scan()["accepted"]), 1)

    def test_environment_mismatch_leaves_ratios_empty(self):
        self.commit()
        def change(reports):
            for report in reports.values():
                report["environment"]["gpu"] = "different GPU"
        self.commit("ort_cuda_fp16", change=change)
        quality, timing, _ = tables(self.scan(), "torch_cuda_fp32")
        current = next(r for r in timing if r["mode"] == "ort_cuda_fp16")
        self.assertEqual(current["baseline_status"], "protocol_mismatch")
        self.assertNotIn("speedup_vs_baseline", current)
        self.assertNotIn("map50_95_delta_pp", quality[1])

    def test_invalid_rounds_validation_counts_and_hashes(self):
        for change in (lambda reports: reports["model_compute"]["rounds"].pop(),
                       lambda reports: reports["evaluation"].update(evaluated_images=2),
                       lambda reports: reports["model_compute"]["rounds"][0].update(images_per_second=99)):
            self.commit(change=change)
            self.assertEqual(self.scan()["routes"][0]["status"], "invalid")
        self.commit()
        latest = read_json(self.stage() / "latest.json")
        latest["artifact_sha256"]["evaluation.json"] = None
        write_json(self.stage() / "latest.json", latest)
        self.assertEqual(self.scan()["routes"][0]["status"], "invalid")

    def test_output_never_overwrites_experiments_and_snapshots_are_unique(self):
        for target in (self.run, self.run / "comparison", self.root / "outputs" / "elsewhere", self.run.parent):
            with self.assertRaises(ValueError):
                snapshot_directory(self.run, target, self.root / "outputs")
        a = snapshot_directory(self.run, self.root / "analysis", self.root / "outputs")
        b = snapshot_directory(self.run, self.root / "analysis", self.root / "outputs")
        self.assertNotEqual(a, b)

    def test_copied_run_output_name_collision_is_rejected(self):
        with self.assertRaises(ValueError):
            snapshot_directory(self.run, self.run.parent, self.root / "different-default-outputs")

    def test_read_handle_allows_atomic_writer_replace(self):
        path = self.root / "live.json"
        path.write_bytes(b'old')
        with open_snapshot(path, self.root) as stream:
            replacement = self.root / "new.json"
            replacement.write_bytes(b'new')
            os.replace(replacement, path)
            self.assertEqual(stream.read(), b'old')
        self.assertEqual(path.read_bytes(), b'new')
        self.assertEqual(list(self.root.glob('.analysis-read-*')), [])

    def test_dependency_failure_survives_old_attempts(self):
        self.commit()
        write_json(self.stage() / "latest.json", {"status": "dependency_failed", "error": "Export failed"})
        snapshot = self.scan()
        self.assertFalse(snapshot["accepted"])
        self.assertEqual(snapshot["routes"][0]["status"], "dependency_failed")
        self.assertEqual(snapshot["routes"][0]["error"], "Export failed")

    def test_new_retry_supersedes_old_dependency_failure(self):
        self.commit()
        latest = self.stage() / "latest.json"
        write_json(latest, {"status": "dependency_failed", "error": "Previous export failed"})
        os.utime(latest, (100, 100))
        newer = self.stage() / "attempt_002"
        write_json(newer / "job.json", {"kind": "route"})
        write_json(newer / "status.json", {"status": "running", "phase": "full_validation"})
        snapshot = self.scan()
        self.assertEqual(snapshot["routes"][0]["status"], "incomplete")
        self.assertTrue(snapshot["routes"][0]["attempt"].endswith("attempt_002"))
        self.assertFalse(snapshot["accepted"])

    def test_malformed_reports_do_not_abort_other_routes(self):
        self.commit()
        self.commit("ort_cuda_fp16")
        latest = read_json(self.stage() / "latest.json")
        latest["artifact_sha256"] = None
        write_json(self.stage() / "latest.json", latest)
        snapshot = self.scan()
        self.assertEqual(len(snapshot["accepted"]), 1)
        self.assertEqual(snapshot["routes"][0]["status"], "invalid")
        self.commit(change=lambda reports: reports["evaluation"].update(environment=None))
        self.assertEqual(len(self.scan()["accepted"]), 1)
        for invalid in ([], {}, "unknown"):
            write_json(self.stage() / "latest.json", {"status": invalid})
            snapshot = self.scan()
            self.assertEqual(snapshot["routes"][0]["status"], "invalid")
            self.assertEqual(len(snapshot["accepted"]), 1)

    def test_latest_run_uses_config_time_not_progress_mtime(self):
        write_json(self.root / "outputs" / "malformed" / "config.json", {"created_at": None})
        newer = self.root / "outputs" / "another-name"
        config = deepcopy(self.config)
        config["created_at"] = "2099-01-01T00:00:00+00:00"
        write_json(newer / "config.json", config)
        os.utime(self.run, None)
        self.assertEqual(select_run(self.root / "outputs"), newer)
        self.assertEqual(select_run(self.root / "outputs", self.run), self.run)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib required for rendering")
    def test_direct_launcher_from_other_cwd_keeps_input_unchanged_and_imports_no_runtime(self):
        self.commit()
        self.commit("ort_cuda_fp16", rate=20, quality=.79)
        before = {p.relative_to(self.run).as_posix(): digest(p) for p in self.run.rglob("*") if p.is_file()}
        code = ("import runpy,sys; "
                f"m=runpy.run_path({str(ROOT / 'analyze_results.py')!r}); "
                f"result=m['main'](['--run',{str(self.run)!r},'--output-root',{str(self.root / 'analysis')!r},"
                "'--formats','png','pdf','--dpi','72']); "
                "assert result == 0; assert not any(n in sys.modules for n in ('torch','onnxruntime','tensorrt'))")
        result = subprocess.run([sys.executable, "-E", "-B", "-c", code], cwd=self.root,
                                text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        after = {p.relative_to(self.run).as_posix(): digest(p) for p in self.run.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        output = next((self.root / "analysis" / self.run.name).iterdir())
        summary = read_json(output / "summary.json")
        self.assertEqual(summary["valid_routes"], 2)
        self.assertTrue((output / "index.html").is_file())
        for figure in summary["figures"]:
            for relative in figure["files"]:
                self.assertGreater((output / relative).stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
