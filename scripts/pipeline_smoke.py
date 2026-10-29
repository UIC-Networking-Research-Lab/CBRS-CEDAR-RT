"""Exercise demo, quality metrics and all timing scopes with an explicit small subset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402
from cbrs_quant.compare import compare  # noqa: E402
from cbrs_quant.data import Dataset  # noqa: E402
from cbrs_quant.io import write_json  # noqa: E402
from cbrs_quant.runner import benchmark, evaluate, predict  # noqa: E402
from cbrs_quant.runtime import TorchBackend  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = Dataset(args.data, args.manifest, "test")
    dataset.records = dataset.records[:8]
    paths = [dataset.path(r["image"]) for r in dataset.records[:3]]
    for model_id in ("cbrs_yolo8s", "sig_yolo_v8s_f20t20_c100"):
        scope_reports = {s: [] for s in ("model_compute", "detector_pipeline", "image_to_detection")}
        for mode in ("torch_cuda_fp32", "torch_cuda_fp16"):
            with torch.cuda.stream(torch.cuda.Stream()):
                backend = TorchBackend(model_id, root / "artifacts/weights" / f"{model_id}.pt", mode, 1)
                destination = args.output / model_id / mode
                predict(backend, paths, destination / "demo")
                quality = evaluate(backend, dataset)
                quality["smoke_only_subset"] = True
                write_json(destination / "evaluation.json", quality)
                for scope in scope_reports:
                    report = benchmark(backend, dataset, scope, warmup=2, min_images=4,
                                       seconds=.1, rounds=1, pool_size=4)
                    report["smoke_only_subset"] = True
                    path = destination / f"{scope}.json"
                    write_json(path, report)
                    scope_reports[scope].append(path)
                del backend
                torch.cuda.empty_cache()
            print(model_id, mode, "pipeline PASS", flush=True)
        for scope, reports in scope_reports.items():
            compare(reports, args.output / model_id / "comparison" / scope)


if __name__ == "__main__":
    main()
