"""Exercise real ORT sessions for all four detector families; no simulated providers."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402
from cbrs_quant.io import write_json  # noqa: E402
from cbrs_quant.runtime import create_backend, is_cpu_mode  # noqa: E402
from cbrs_quant.smoke import smoke  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["ort_cpu_fp32", "ort_cuda_fp32"])
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    paths = sorted((root / "data/yolo_split/test/images").glob("*.png"))[:3]
    if not paths:
        raise FileNotFoundError("No test PNGs found under data/yolo_split/test/images")
    reports = []
    for model_id in ("cbrs_yolo8n", "cbrs_yolo11n", "cbrs_yolo26n", "sig_yolo_v8n_f20t20_c100"):
        for mode in args.modes:
            suffix = "fp16" if mode == "ort_cuda_fp16" else ("int8" if mode.endswith("int8") else "fp32")
            context = nullcontext() if is_cpu_mode(mode) else torch.cuda.stream(torch.cuda.Stream())
            with context:
                weights = root / "artifacts/weights" / f"{model_id}.pt"
                backend = create_backend(mode, model_id, weights, None, onnx=args.onnx_dir / f"{model_id}.{suffix}.onnx",
                                         cache_dir=args.output / "cache", allow_cpu_fallback=args.allow_cpu_fallback)
                report = smoke(backend, paths, weights, max_relative_l2=.01 if suffix == "fp32" else .05)
                reports.append(report)
                write_json(args.output / "summary.json", reports)
                print(model_id, mode, "PASS", backend.meta["provider_audit"]["node_events"], flush=True)
                del backend
    return reports


if __name__ == "__main__":
    main()
