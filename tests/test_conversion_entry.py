"""Conversion entry tests; use synthetic tensors and temporary files."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("conversion_entry", ROOT / "export_onnx.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class ConversionEntryTests(unittest.TestCase):
    def test_default_plan_is_dynamic_and_cli_can_choose_static(self):
        self.assertIsNone(entry.parser().parse_args([]).batch)
        self.assertEqual(entry.parser().parse_args(["--batch", "4"]).batch, 4)
        with patch.object(entry, "BATCH_SIZE", 4):
            self.assertIsNone(entry.parser().parse_args(["--dynamic"]).batch)

    def test_dry_run_does_not_import_runtime_or_write_files(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as stream:
            target = Path(tmp) / "model.onnx"
            weights = Path(tmp) / "best.pt"  # not read in a dry run
            self.assertEqual(entry.main(["--weights", str(weights), "--output", str(target), "--dry-run"]), 0)
            self.assertIn("dynamic", stream.getvalue())
            self.assertFalse(target.exists())

    def test_invalid_defaults_and_extension_are_rejected(self):
        with patch.object(entry, "PRECISION", "bad"), redirect_stderr(io.StringIO()):
            self.assertEqual(entry.main(["--batch", "4"]), 1)
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(["--weights", "best.pt", "--output", "invalid.txt", "--dry-run"]), 1)
            self.assertEqual(entry.main(["--model", "../escape", "--dry-run"]), 1)

    def test_manifest_weight_digest_is_checked(self):
        from cbrs_quant.io import write_json
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            folder = Path(tmp)
            checkpoint = folder / "weights.pt"
            checkpoint.write_bytes(b"changed")
            manifest = folder / "manifest.json"
            write_json(manifest, [{"model_id": "cbrs_yolo8n", "path": str(checkpoint), "sha256": "0" * 64}])
            self.assertEqual(entry.main(["--weights-manifest", str(manifest), "--output", str(folder / "out.onnx")]), 1)
            self.assertFalse((folder / "out.onnx").exists())

    def test_dynamic_graph_handles_multiple_batches(self):
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from cbrs_quant.dynamic_export import export_dynamic
        from cbrs_quant.io import read_json

        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Identity()
            def forward(self, images):
                return images.mean(dim=(2, 3)).unsqueeze(-1)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "model.onnx"
            with patch("cbrs_quant.dynamic_export.load_model", return_value=(Detector(), {"model_id": "fixture"})), \
                    patch("cbrs_quant.dynamic_export.environment", return_value={}):
                meta = export_dynamic("fixture", Path(tmp) / "unused.pt", output)
            self.assertTrue(meta["batch_dynamic"])
            self.assertEqual(read_json(Path(str(output) + ".json"))["output_shape"], ["batch", 3, 1])
            graph = onnx.load(str(output))
            self.assertEqual(graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
            for batch in (1, 4):
                images = np.ones((batch, 3, 640, 640), dtype=np.float32)
                result = session.run(None, {"images": images})[0]
                self.assertEqual(result.shape, (batch, 3, 1))
                np.testing.assert_allclose(result, 1)


if __name__ == "__main__":
    unittest.main()
