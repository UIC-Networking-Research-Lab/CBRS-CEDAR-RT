"""Export one small checkpoint per architecture; four calibration PNGs test plumbing only."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402
from cbrs_quant.data import Dataset  # noqa: E402
from cbrs_quant.export import export_onnx  # noqa: E402
from cbrs_quant.io import write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = Dataset(args.data, args.manifest, "train")
    reports = []
    for name in ("cbrs_yolo8n", "cbrs_yolo11n", "cbrs_yolo26n", "sig_yolo_v8n_f20t20_c100"):
        for precision in ("fp32", "int8"):
            output = args.output / f"{name}.{precision}.onnx"
            meta = export_onnx(name, root / "artifacts/weights" / f"{name}.pt", output,
                               dataset=dataset if precision == "int8" else None, samples=4)
            reports.append({"artifact": str(output), "qdq_nodes": meta["qdq_nodes"],
                            "output_shape": meta["output_shape"], "sha256": meta["sha256"]})
            write_json(args.output / "summary.json", reports)
            print(name, precision, "PASS", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
