"""Preserve recorded values and detect corruption when consolidating experiments."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from cbrs_quant.results_store import export_data, included, load_data, seal, table, digest
from cbrs_quant.rt_report import enrich, generate


class ResultsStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output=Path(self.tmp.name)/"results"
        self.data=self.output/"data"
        meta={"model_id":"cbrs_yolo8n","experiment":{"name":"fixture","model_order":["cbrs_yolo8n"],
              "batches":[1,4],"modes":["torch_cuda_fp32","ort_cpu_fp32"],"protocol":{"rounds":3,"evaluation_split":"val"}},
              "routes":{"1:torch_cuda_fp32":{"status":"passed"}},"environments":{},"dataset":{"records_sha256":"abc"}}
        raw=[{"model":"cbrs_yolo8n","batch":"1","mode":"torch_cuda_fp32","status":"passed","scope":"model_compute",
              "round":str(i+1),"images_per_second":str(rate),"latency_ms_p50":"8","latency_ms_p95":"9","latency_ms_p99":"10",
              "map50":".95","map50_95":".8","lte_f1":".7","radar_f1":".9","radar_recall":".9",
              "timing_signature":"timing","quality_signature":"quality","evaluated_images":"846"} for i,rate in enumerate((100,200,300))]
        self.bundle={"models":{"cbrs_yolo8n":{"meta":meta,"raw":raw}},"order":["cbrs_yolo8n"],"source":[],"run_name":"fixture",
                     "report_scope":{"models":["cbrs_yolo8n"],"batches":[1],"modes":["torch_cuda_fp32"]}}
        export_data(self.bundle,self.data)
        seal(self.data)

    def test_roundtrip_preserves_raw_rows_and_aggregates(self):
        loaded=load_data(self.output)
        original=self.bundle["models"]["cbrs_yolo8n"]
        self.assertEqual(loaded["models"]["cbrs_yolo8n"]["raw"],original["raw"])
        self.assertEqual(loaded["models"]["cbrs_yolo8n"]["rows"],enrich(original["meta"],original["raw"]))
        self.assertEqual(table(self.output,"rounds"),original["raw"])

    def test_checksum_rejects_changed_measurements(self):
        p=self.data/"rounds.csv"
        p.write_text(p.read_text(encoding="utf-8").replace("100,","101,"),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"checksum mismatch"):
            load_data(self.output)

    def test_scope_excludes_canceled_cpu_and_unrequested_batches(self):
        scope=self.bundle["report_scope"]
        self.assertTrue(included({"model":"cbrs_yolo8n","batch":1,"mode":"torch_cuda_fp32"},scope))
        self.assertFalse(included({"model":"cbrs_yolo8n","batch":1,"mode":"ort_cpu_fp32"},scope))
        self.assertFalse(included({"model":"cbrs_yolo8n","batch":4,"mode":"torch_cuda_fp32"},scope))

    def test_manifest_cannot_escape_data_directory(self):
        p=self.data/"manifest.json"
        manifest=json.loads(p.read_text(encoding="utf-8"))
        manifest["files"][0]["path"]="../outside.json"
        p.write_text(json.dumps(manifest),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"escapes"):
            load_data(self.output)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"),"matplotlib required")
    def test_in_place_report_regeneration_keeps_source_data_unchanged(self):
        names=("metadata.json","rounds.csv","history.jsonl")
        before={name:digest(self.data/name) for name in names}
        result=generate(self.data,self.output,formats=("png",),dpi=72,scope="model_compute")
        self.assertEqual(result["comparisons"],1)
        self.assertEqual(before,{name:digest(self.data/name) for name in names})
        self.assertTrue((self.output/"REPORT.md").is_file())
        self.assertTrue((self.data/"summary.csv").is_file())
        self.assertFalse((self.output/"source").exists())
        self.assertFalse((self.output/"comparison").exists())
        load_data(self.output)


if __name__=="__main__":
    unittest.main()
