"""CPU unit tests for contracts and metric edge cases; no inference CPU benchmark."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cbrs_quant.data import Dataset, load_labels, load_png, make_manifest
from cbrs_quant.io import fingerprint, read_json, reserve_output, write_json
from cbrs_quant.metrics import match, summarize, frequency_overlap, overlap_summary


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_strict_png(self):
        path = self.root / "image.png"
        Image.new("RGBA", (640, 640), (5, 8, 9, 255)).save(path)
        self.assertEqual(load_png(path).shape, (640, 640, 3))
        Image.new("RGB", (320, 320)).save(path)
        with self.assertRaises(ValueError):
            load_png(path)
        Image.new("RGBA", (640, 640), (5, 8, 9, 20)).save(path)
        with self.assertRaises(ValueError):
            load_png(path)

    def test_label_validation(self):
        path = self.root / "labels.txt"
        path.write_text("1 .5 .5 .25 .5\n")
        np.testing.assert_allclose(load_labels(path), [[1, 240, 160, 400, 480]])
        for invalid in ("2 .5 .5 .2 .2", "0 nan .5 .2 .2", "0 .9 .5 .5 .5"):
            path.write_text(invalid)
            with self.assertRaises(ValueError):
                load_labels(path)

    def test_rgb_transparency_rejected(self):
        path = self.root / "transparent.png"
        Image.new("RGB", (640, 640), (1, 2, 3)).save(path, transparency=(1, 2, 3))
        with self.assertRaises(ValueError):
            load_png(path)

    def dataset(self):
        profile = self.root / "profile.json"
        write_json(profile, {"image_size": [640, 640], "classes": ["LTE", "Radar"]})
        for i, split in enumerate(("train", "val", "test")):
            images, labels = self.root / split / "images", self.root / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            Image.new("RGB", (640, 640), (i, 0, 0)).save(images / f"{split}.png")
            (labels / f"{split}.txt").write_text("")
        manifest = self.root / "manifest.json"
        make_manifest(self.root, profile, manifest)
        return manifest

    def test_tampering_and_split(self):
        manifest = self.dataset()
        self.assertEqual(len(Dataset(self.root, manifest, "train").calibration(1, 42)), 1)
        with self.assertRaises(ValueError):
            Dataset(self.root, manifest, "test").calibration(1, 42)
        (self.root / "train" / "labels" / "train.txt").write_text("0 .5 .5 .2 .2")
        with self.assertRaises(ValueError):
            Dataset(self.root, manifest, "train")

    def test_manifest_path_escape(self):
        path = self.dataset()
        manifest = read_json(path)
        manifest["records"][0]["image"] = "../outside.png"
        manifest["records_sha256"] = fingerprint(manifest["records"])
        write_json(path, manifest)
        with self.assertRaises(ValueError):
            Dataset(self.root, path, "train", verify=False)

    def test_atomic_json_and_immutable_artifact(self):
        path = self.root / "report.json"
        write_json(path, {"a": 1})
        self.assertEqual(read_json(path), {"a": 1})
        with self.assertRaises(FileExistsError):
            reserve_output(path)


class Metrics(unittest.TestCase):
    def test_overlap_ignores_time_but_requires_positive_frequency_width(self):
        boxes = np.array([[0, 0, 10, 1], [5, 50, 15, 60]], dtype=float)
        self.assertTrue(frequency_overlap(boxes, np.array([0, 1])))
        boxes[1, 0] = 10
        self.assertFalse(frequency_overlap(boxes, np.array([0, 1])))
        self.assertFalse(frequency_overlap(boxes, np.array([0, 0])))

    def test_overlap_pools_frame_counts_not_box_f1(self):
        positive = np.array([[0, 0, 0, 10, 10], [1, 5, 50, 15, 60]], dtype=float)
        negative = positive.copy()
        negative[1, 1] = 10
        pred = np.array([[0, 0, 10, 10, .9, 0], [5, 50, 15, 60, .9, 1]])
        low_conf = pred.copy()
        low_conf[1, 4] = .1
        empty = np.empty((0, 6))
        result = overlap_summary([(pred, positive), (pred, negative),
                                  (low_conf, positive), (empty, negative)])
        self.assertEqual([result[k] for k in ('TP', 'TN', 'FP', 'FN')], [1, 1, 1, 1])
        self.assertAlmostEqual(result['f1'], .5)

    def test_operating_confidence_must_be_collected(self):
        for confidence in (-1, 0, .0001, 2):
            with self.assertRaises(ValueError):
                summarize([(np.empty((0, 6)), np.empty((0, 5)))], confidence)

    def test_perfect_and_missing_class(self):
        pred = np.array([[0, 0, 10, 10, .9, 1]])
        truth = np.array([[1, 0, 0, 10, 10]])
        result = summarize([(pred, truth)])
        self.assertEqual(result["map50_95"], 1)
        self.assertIsNone(result["classes"]["LTE"]["ap50"])
        self.assertEqual(result["classes"]["Radar"]["recall"], 1)

    def test_duplicates_and_wrong_class(self):
        pred = np.array([[0, 0, 10, 10, .9, 1], [0, 0, 10, 10, .8, 1],
                         [0, 0, 10, 10, .7, 0]])
        truth = np.array([[1, 0, 0, 10, 10]])
        self.assertEqual(int(match(pred, truth)[:, 0].sum()), 1)
        result = summarize([(pred, truth)])
        self.assertEqual(result["classes"]["Radar"]["false_positives"], 1)

    def test_empty_predictions_and_false_positive_image(self):
        empty = np.empty((0, 6))
        truth = np.array([[1, 0, 0, 10, 10]])
        result = summarize([(empty, truth)])
        self.assertEqual(result["map50_95"], 0)
        self.assertEqual(result["classes"]["Radar"]["recall"], 0)


if __name__ == "__main__":
    unittest.main()
