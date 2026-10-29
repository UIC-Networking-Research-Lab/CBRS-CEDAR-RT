"""Check backend-independent decoding and Q/DQ mechanics on tiny tensors."""

import unittest

import torch

from cbrs_quant.export import QDQConv
from cbrs_quant.postprocess import detections


class TorchContracts(unittest.TestCase):
    def test_calibration_profile_mismatch_rejected(self):
        from types import SimpleNamespace
        from cbrs_quant.runner import validate_dataset_backend
        backend = SimpleNamespace(meta={"calibration": {"profile_sha256": "a",
                                                        "dataset_sha256": "same"}})
        data = SimpleNamespace(manifest={"profile_sha256": "b", "records_sha256": "same"})
        with self.assertRaises(ValueError):
            validate_dataset_backend(backend, data)

    def test_negative_coordinates_do_not_cross_suppress(self):
        raw = torch.tensor([[[-500, -500], [-500, -500], [100, 100], [100, 100],
                             [.9, .1], [.1, .8]]])
        self.assertEqual(len(detections(raw, "xyxy_scores")[0]), 2)

    def test_topk_permutation_parity(self):
        import numpy as np
        from cbrs_quant.smoke import end2end_parity
        boxes = np.array([[0, 0, 10, 10, .9, 1], [30, 30, 50, 50, .8, 0]])
        self.assertTrue(end2end_parity(boxes, boxes[::-1]))
        self.assertFalse(end2end_parity(boxes, boxes[:1]))

    def test_demo_cannot_overwrite_input(self):
        import tempfile
        from pathlib import Path
        from cbrs_quant.runner import validate_demo_output
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.png"
            path.touch()
            with self.assertRaises(FileExistsError):
                validate_demo_output([path], Path(folder))

    def test_class_aware_nms(self):
        raw = torch.tensor([[[0, 0, 0], [0, 0, 0], [10, 10, 10], [10, 10, 10],
                             [.9, .8, .1], [.1, .1, .9]]])
        result = detections(raw, "xyxy_scores")[0]
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result[:, 5].tolist()), {0, 1})

    def test_end2end_is_not_suppressed(self):
        raw = torch.tensor([[[0, 0, 10, 10, .9, 1], [0, 0, 10, 10, .8, 1]]])
        self.assertEqual(len(detections(raw, "end2end")[0]), 2)

    def test_xywh_conversion(self):
        raw = torch.tensor([[[5], [5], [10], [10], [.9], [.1]]])
        torch.testing.assert_close(detections(raw, "xywh_scores")[0][0, :4],
                                   torch.tensor([0., 0, 10, 10]))

    def test_qdq_calibration(self):
        torch.manual_seed(1)
        conv = torch.nn.Conv2d(3, 4, 3).eval()
        wrapper = QDQConv(conv)
        inputs = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            original = wrapper(inputs)
            wrapper.freeze()
            quantized = wrapper(inputs)
        self.assertGreater(wrapper.input_scale, 0)
        self.assertTrue(torch.isfinite(quantized).all())
        self.assertLess(float((quantized-original).abs().max()), .04)


if __name__ == "__main__":
    unittest.main()
