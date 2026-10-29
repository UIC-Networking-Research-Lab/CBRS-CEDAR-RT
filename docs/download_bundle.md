# CBRS-CEDAR-RT assets

Data and pretrained models for the **CBRS-CEDAR-RT GPU deployment experiments**.
This download is independent of the MDI paper project's asset package.

## Download and extract

**[Download CBRS-CEDAR-RT-ASSETS from GitHub Releases](https://github.com/UIC-Networking-Research-Lab/CBRS-CEDAR-RT/releases/tag/assets)**

1. Open the release's **Assets** list and download **all 3 archive parts**:
   `CBRS-CEDAR-RT-ASSETS.7z.001` through `CBRS-CEDAR-RT-ASSETS.7z.003`.
   Keep all parts in the same folder with their original names.
2. Use **7-Zip** to extract `CBRS-CEDAR-RT-ASSETS.7z.001` only. It reads the remaining
   parts automatically; do not extract each part separately.
3. Place the extracted `CBRS-CEDAR-RT-ASSETS` folder beside the repository:

```text
workspace/
  CBRS-CEDAR-RT/
  CBRS-CEDAR-RT-ASSETS/
    datasets/
    weights/
    splits.csv
    README.md
```

Alternatively, extract from the folder containing all parts with:

```bash
7z x CBRS-CEDAR-RT-ASSETS.7z.001
```

GitHub's **Source code (zip/tar.gz)** downloads contain code, not the datasets or
weights. Use this project's asset package; the MDI and RT packages are independent.

## Package contents

```text
CBRS-CEDAR-RT-ASSETS/
  datasets/
    train/{images,labels}/
    val/{images,labels}/
    test/{images,labels}/
  weights/
    cbrs_yolo*.pt
    sig_yolo_v8*_f20t20_c100.pt
  splits.csv
  README.md
```

Extract this folder beside the CBRS-CEDAR-RT repository. From the repository root:

```bash
python reproduce.py --assets ../CBRS-CEDAR-RT-ASSETS --task install
python reproduce.py --task check
python reproduce.py --task smoke
python reproduce.py --task full
```

`datasets/` contains all prepared **640 × 640 PNGs and YOLO labels**: 3,946 training,
846 validation and 848 test samples. Class 0 is LTE; class 1 is radar.
**The recorded RT experiments evaluate the validation split**, with calibration
images drawn from the training split. `splits.csv` preserves the exact membership.
The test split is also included for further experiments.

`weights/` contains the **12 reference checkpoints**: YOLOv8, YOLO11 and YOLO26
in nano/small/extra-large sizes, plus three CEDAR-YOLO models with a 20 × 20 grid
and 100% regression coverage. The CEDAR checkpoints come from the latest local June model collection.
The recorded RT results use the earlier edition; the new models have not yet been benchmarked.
The checkpoint edition does not identify the execution platform: **all RT results
were measured on the PC GPU**, with the software versions listed in the repository README; there are no Jetson
deployment experiments in this report. Tested PC versions are listed in the repository README.

The installer places data in `data/yolo_split/` and weights in `artifacts/weights/`.
Use the supplied PNGs directly; GPU inference and calibration need no raw I/Q.
The [Genesys Lab dataset page](https://genesys-lab.org/CBRS) provides the original
over-the-air radar/LTE recordings. Follow its terms and cite the dataset paper.
Raw recordings are not redistributed here and are unnecessary for the RT benchmarks.

**The original Genesys frames do not include our train/validation/test partitions.**
If you download raw recordings to regenerate PNGs, use the assets' `splits.csv`
(or the repository's [`splits/cbrs_split.csv`](../splits/cbrs_split.csv)) to restore
the exact split membership. Follow the
[MDI raw-data preparation guide](https://github.com/UIC-Networking-Research-Lab/CBRS-CEDAR-MDI/blob/main/docs/data.md#5-build-the-raw-sigmf-split),
and preserve that membership through conversion. Use training images for calibration
and validation images for reproducing the RT evaluation. The PNGs in this download
already follow these splits.

**Preprocessing parameters and library versions change the spectrogram pixels.
Even a small mismatch can invalidate predictions from pretrained weights.** Use the
supplied PNGs directly, or preserve rendering versions/settings and verify pixel
equivalence before using regenerated images. For a different preprocessing pipeline,
retrain your model and use the same rendering for training, calibration and deployment.
The checked renderer uses NumPy 2.2.4, SciPy 1.14.1, Matplotlib 3.10.8 and Pillow
11.2.1. Rendering parameters are documented in the repository README.

The repository contains all scripts, configuration and experiment reports.
This download contains only datasets, weights, the split list and this README.

## Updated CEDAR checkpoints

Use the updated asset volumes containing the latest local June CEDAR models. Nano and small have changed; extra-large and all original YOLO weights are unchanged. Do not mix volumes from different asset releases. The updated installer upgrades recognized old CEDAR weights. Start a new experiment directory and export new ONNX/TensorRT artifacts. Published results still describe the earlier checkpoints and box-detection F1 until a new sweep is completed.
