"""Completion plans must include pending work without repeating successful routes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("complete_batch_sweep", Path(__file__).resolve().parents[1]/"scripts/complete_batch_sweep.py")
completion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        model = self.run/"example"
        model.mkdir()
        data = {"model_id": "example", "experiment": {"model_order": ["example"],
            "batches": [1,4,8], "modes": ["torch_cuda_fp32","ort_cpu_fp32"]},
            "routes": {"1:torch_cuda_fp32": {"status": "passed"},
                       "4:torch_cuda_fp32": {"status": "failed"},
                       "8:torch_cuda_fp32": {"status": "pending"}}}
        (model/"model.json").write_text(json.dumps(data),encoding="utf-8")

    def test_gpu_only_keeps_failed_and_pending_skips_success(self):
        result = completion.pending(self.run,[1,4,8])
        self.assertEqual([(r["batch"],r["reason"]) for r in result],[(4,"failed"),(8,"pending")])

    def test_cpu_work_is_added_after_gpu_work(self):
        result = completion.pending(self.run,[1,4,8],include_cpu=True)
        self.assertEqual(len(result),5)
        self.assertEqual([r["mode"] for r in result[-3:]], ["ort_cpu_fp32"]*3)
        self.assertEqual([r["batch"] for r in result[-3:]], [1,4,8])


if __name__ == "__main__":
    unittest.main()
