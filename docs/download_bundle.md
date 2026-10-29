# CBRS-CEDAR-RT assets

Data and pretrained models for the **CBRS-CEDAR-RT GPU deployment experiments**.
This download is independent of the MDI paper project's asset package.

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
and 100% regression coverage. The CEDAR checkpoints are the earlier Jetson edition
used in the recorded RT results. Later local/sweep weights belong to the MDI package.
The checkpoint edition does not identify the execution platform: **all RT results
were measured on the PC GPU**, in Conda `torch270_py312_cuda`; there are no Jetson
deployment experiments in this report. Tested PC versions are listed in the repository README.

The installer places data in `data/yolo_split/` and weights in `artifacts/weights/`.
Use the supplied PNGs directly; GPU inference and calibration need no raw I/Q.
The [Genesys Lab dataset page](https://genesys-lab.org/CBRS) provides the original
over-the-air radar/LTE recordings. Follow its terms and cite the dataset paper.
Raw recordings are not redistributed here.

**Preprocessing parameters and library versions change the spectrogram pixels.
Even a small mismatch can invalidate predictions from pretrained weights.** Use the
supplied PNGs directly, or preserve rendering versions/settings and verify pixel
equivalence before using regenerated images. For a different preprocessing pipeline,
retrain your model and use the same rendering for training, calibration and deployment.
The checked renderer uses NumPy 2.2.4, SciPy 1.14.1, Matplotlib 3.10.8 and Pillow
11.2.1. Rendering parameters are documented in the repository README.

The repository contains all scripts, configuration and experiment reports.
This download contains only datasets, weights, the split list and this README.
