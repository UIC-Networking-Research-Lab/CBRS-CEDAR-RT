"""Meaningful checks for report metrics and portable experiment records."""
from pathlib import Path
import tempfile
import unittest

from cbrs_quant.rt_report import enrich, load_run, portable_source, sanitize


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.meta = {"model_id": "cbrs_yolo8n", "experiment": {
            "name": "test-run", "created_at": "2026-10-28T00:00:00Z",
            "model_order": ["cbrs_yolo8n"], "batches": [1], "modes": ["torch_cuda_fp32"],
            "protocol": {"rounds": 3, "evaluation_split": "val"}},
            "routes": {}, "dataset": {"records_sha256": "abc"}}
        self.rows = [{"model": "cbrs_yolo8n", "batch": "1", "mode": "torch_cuda_fp32",
            "status": "passed", "scope": "model_compute", "round": str(i+1),
            "images_per_second": str(rate), "latency_ms_p50": "8", "latency_ms_p95": "9",
            "latency_ms_p99": "10", "map50": ".95", "map50_95": ".8",
            "lte_f1": ".7", "radar_f1": ".9", "radar_recall": ".9",
            "timing_signature": "timing", "quality_signature": "quality", "evaluated_images": "846"}
            for i, rate in enumerate((100, 200, 300))]

    def test_inverse_of_mean_throughput_and_macro_detection_f1(self):
        result = enrich(self.meta, self.rows)[0]
        self.assertAlmostEqual(result["inference_time_per_frame_s"], .005)
        self.assertAlmostEqual(result["macro_f1"], .8)
        self.assertEqual(result["evaluation_split"], "val")

    def test_mixed_protocols_are_rejected_before_averaging(self):
        self.rows[-1]["timing_signature"] = "different"
        with self.assertRaisesRegex(ValueError, "mixed timing/quality"):
            enrich(self.meta, self.rows)

    def test_duplicate_rounds_are_rejected(self):
        self.rows[-1]["round"] = "2"
        with self.assertRaisesRegex(ValueError, "unique timing rounds"):
            enrich(self.meta, self.rows)

    def test_changing_quality_is_not_silently_averaged(self):
        self.rows[-1]["lte_f1"] = ".4"
        with self.assertRaisesRegex(ValueError, "inconsistent lte_f1"):
            enrich(self.meta, self.rows)

    def test_missing_class_f1_is_not_fabricated(self):
        for row in self.rows:
            row["radar_f1"] = ""
        self.assertIsNone(enrich(self.meta, self.rows)[0]["macro_f1"])

    def test_overlap_is_separate_from_box_f1(self):
        for row in self.rows:
            row.update(overlap_f1='0.99', overlap_TP='495', overlap_FP='5',
                       overlap_FN='5', overlap_TN='341')
        result = enrich(self.meta, self.rows)[0]
        self.assertAlmostEqual(result['overlap_f1'], .99)
        self.assertAlmostEqual(result['macro_f1'], .8)

    def test_missing_overlap_is_not_replaced_with_detection_f1(self):
        self.assertIsNone(enrich(self.meta, self.rows)[0]['overlap_f1'])

    def test_public_bundle_reproduces_aggregate_values(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report"
            bundle = {"models": {"cbrs_yolo8n": {"meta": self.meta, "raw": self.rows}},
                      "run_name": "test-run", "source": []}
            portable_source(bundle, output)
            loaded = load_run(output / "source")
            self.assertEqual(loaded["models"]["cbrs_yolo8n"]["rows"], enrich(self.meta, self.rows))

    def test_redaction_preserves_metric_values_and_error_meaning(self):
        row = {"error": r"Access denied C:\Users\someone\private\model.onnx", "f1": ".99999"}
        clean = sanitize(row)
        self.assertNotIn("someone", clean["error"])
        self.assertIn("Access denied", clean["error"])
        self.assertEqual(clean["f1"], row["f1"])
        self.assertIn("someone", row["error"])


if __name__ == "__main__":
    unittest.main()
