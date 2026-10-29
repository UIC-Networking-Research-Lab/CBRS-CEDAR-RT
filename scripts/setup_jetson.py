"""Install pipeline dependencies while constraining NVIDIA's existing GPU wheels."""

import importlib.metadata
import argparse
import platform
import zipfile
from email.parser import Parser
import subprocess
import sys
import tempfile
from pathlib import Path


def wheel_distribution(path: Path) -> str:
    with zipfile.ZipFile(path) as wheel:
        names = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("Wheel must have one distribution metadata file")
        name = Parser().parsestr(wheel.read(names[0]).decode("utf-8"))["Name"]
    return name.lower().replace("_", "-") if name else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ort-wheel", type=Path, help="GPU wheel matched to Jetson Python/CUDA/cuDNN/TensorRT")
    parser.add_argument("--ort-cpu-only", action="store_true", help="Install CPU-only ORT for route D")
    args = parser.parse_args()
    if args.ort_wheel and args.ort_cpu_only:
        parser.error("Choose one ORT distribution")
    # A normal resolver may otherwise replace Jetson's custom CUDA wheels.
    protected = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy")}
    with tempfile.TemporaryDirectory(prefix="cbrs-deps-") as folder:
        constraints = Path(folder) / "constraints.txt"
        constraints.write_text("".join(f"{name}=={version}\n" for name, version in protected.items()))
        packages = ["ultralytics==8.4.46", "onnx==1.18.0", "Pillow>=10", "PyYAML>=6"]
        try:
            import onnxruntime as ort
        except ImportError:
            ort = None
        installed = []
        for name in ("onnxruntime", "onnxruntime-gpu"):
            try:
                importlib.metadata.version(name)
                installed.append(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        if len(installed) == 2:
            raise RuntimeError("Both ORT distributions are installed; use a clean environment with one wheel")
        if args.ort_wheel:
            if not args.ort_wheel.is_file():
                raise FileNotFoundError(args.ort_wheel)
            if wheel_distribution(args.ort_wheel) != "onnxruntime-gpu":
                raise ValueError("--ort-wheel requires an onnxruntime-gpu distribution")
            if "onnxruntime" in installed:
                raise RuntimeError("Remove CPU-only ORT before installing a GPU wheel; they share the same module")
            packages.append(str(args.ort_wheel.resolve()))
        elif args.ort_cpu_only:
            if "onnxruntime-gpu" in installed:
                raise RuntimeError("GPU ORT already includes CPU EP; do not install a second distribution")
            packages.append("onnxruntime==1.22.0")
        elif ort is None or "CUDAExecutionProvider" not in ort.get_available_providers():
            if platform.machine().lower() in ("aarch64", "arm64"):
                raise RuntimeError("Provide --ort-wheel matched to this Jetson. See the Installation section of README.md")
            if "onnxruntime" in installed:
                raise RuntimeError("CPU-only ORT is installed; use a clean GPU environment")
            packages.append("onnxruntime-gpu==1.22.0")
        subprocess.run([sys.executable, "-m", "pip", "install", "--constraint", str(constraints),
                        "-r", str(Path(__file__).resolve().parents[1] / "requirements.txt"),
                        *packages], check=True)
    for name, version in protected.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Protected dependency changed: {name}")
    import torch
    import torchvision
    import onnx
    import ultralytics
    print({"torch": torch.__version__, "torchvision": torchvision.__version__,
           "onnx": onnx.__version__, "ultralytics": ultralytics.__version__})
    required = ["CPUExecutionProvider"] if args.ort_cpu_only else ["CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"]
    check = ("import onnxruntime as o; p=o.get_available_providers(); "
             f"required={required!r}; print(o.__version__,p); "
             "assert all(x in p for x in required), f'Missing required ORT providers: {required}; got {p}'")
    subprocess.run([sys.executable, "-c", check], check=True)


if __name__ == "__main__":
    main()
