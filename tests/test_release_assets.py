"""Project-specific bundle installation, without GPU or downloaded assets."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.release_assets import digest, install


class AssetInstallTests(unittest.TestCase):
    def fixture(self, root):
        bundle = root / 'bundle'
        records = []
        for source, destination, data in [
            ('datasets/val/images/sample.png', 'data/yolo_split/val/images/sample.png', b'image fixture'),
            ('weights/example.pt', 'artifacts/weights/example.pt', b'weight fixture'),
            ('splits.csv', 'splits/cbrs_split.csv', b'split,band,stem\nval,10M,sample\n'),
        ]:
            path = bundle / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            records.append({'source': source, 'destination': destination,
                            'bytes': len(data), 'sha256': digest(path)})
        catalog = root / 'repository_assets.json'
        catalog.write_text(json.dumps({'schema': 2, 'project': 'rt', 'files': records,
                                       'dataset': {'records': [{'split': 'val'}]}}))
        return bundle, catalog

    def test_fresh_install_creates_runtime_metadata_without_bundle_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, catalog = self.fixture(root)
            repo = root / 'repo'
            install(bundle, repo, catalog_path=catalog)
            self.assertFalse(list(bundle.rglob('*.json')))
            weights = json.loads((repo / 'artifacts/weights/manifest.json').read_text())
            self.assertEqual(weights[0]['model_id'], 'example')
            self.assertEqual(weights[0]['sha256'], digest(bundle / 'weights/example.pt'))
            self.assertEqual(json.loads((repo / 'artifacts/dataset.json').read_text()),
                             {'records': [{'split': 'val'}]})
            self.assertTrue((repo / 'data/yolo_split/data.yaml').is_file())
            install(bundle, repo, catalog_path=catalog)
            (repo / 'artifacts/weights/example.pt').write_bytes(b'changed')
            with self.assertRaises(FileExistsError):
                install(bundle, repo, catalog_path=catalog)

    def test_bad_asset_fails_before_any_installation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, catalog = self.fixture(root)
            (bundle / 'weights/example.pt').write_bytes(b'bad weights')
            with self.assertRaises(ValueError):
                install(bundle, root / 'repo', catalog_path=catalog)
            self.assertFalse((root / 'repo').exists())

    def test_mdi_catalog_cannot_be_installed_as_rt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, catalog = self.fixture(root)
            data = json.loads(catalog.read_text())
            data['project'] = 'mdi'
            catalog.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                install(bundle, root / 'repo', catalog_path=catalog)
