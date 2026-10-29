"""Copy known checkpoints and a small PNG demo from a local CBRS-CEDAR-MDI checkout.

The source needs trained_models/ (released weights) and yolo_split/ (prepared data).
--cedar-models-dir selects the CEDAR-YOLO checkpoint set; the reference results in
results/ used trained_models_mdi.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cbrs_quant.io import digest, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cedar-models-dir", default="trained_models",
                        help="Folder under --source holding the sig_yolo_* checkpoints")
    args = parser.parse_args()
    records = []
    ids = [f"cbrs_yolo{family}{size}" for family in (8, 11, 26) for size in "nsx"]
    ids += [f"sig_yolo_v8{size}_f20t20_c100" for size in "nsx"]
    for model_id in ids:
        models_dir = args.cedar_models_dir if model_id.startswith("sig_yolo_") else "trained_models"
        source = args.source / models_dir / model_id / "weights" / "best.pt"
        output = args.destination / "artifacts" / "weights" / f"{model_id}.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and digest(output) != digest(source):
            raise FileExistsError(f"Different checkpoint exists: {output}")
        if not output.exists():
            shutil.copyfile(source, output)
        records.append({"model_id": model_id, "path": output.relative_to(args.destination).as_posix(),
                        "sha256": digest(output), "source": source.relative_to(args.source).as_posix()})
    demo = args.destination / "data" / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    for source in sorted((args.source / "yolo_split" / "test" / "images").glob("*.png"))[:3]:
        output = demo / source.name
        if output.exists() and digest(output) != digest(source):
            raise FileExistsError(output)
        shutil.copyfile(source, output)
    write_json(args.destination / "artifacts" / "weights" / "manifest.json", records)
    print(json.dumps({"checkpoints": len(records), "demo_pngs": len(list(demo.glob('*.png')))}))


if __name__ == "__main__":
    main()
