# DropletLab · Microfluidic Droplet Image Training Workbench

[简体中文](README.md) | [English](README.en.md)

A local browser workbench for microfluidic microscopy images: import images, annotate droplet classes, apply batch preprocessing, split datasets, train models, validate predictions and review model outputs. Images and annotations are stored on your computer.

Supports **ConvNeXt-Tiny / ResNet18 single-droplet classification** and **YOLO object detection** for originals containing multiple droplets.

## Features

- Switch between **Chinese and English** using the top-right language selector. The browser remembers the choice. Project names, filenames and model class identifiers retain their original values.
- Organize projects and experiment batches, import PNG/JPG folders, review progress, and preview or restore recycled images.
- Draw droplet boxes or classify a single-droplet ROI with keys **1–5**. Use arrow keys to navigate. Annotation changes save automatically.
- Save reusable crop, grayscale, brightness, contrast and scale rules for batch image processing.
- Compare gentle contrast, illumination and denoising enhancements. Training, validation and prediction share the enhancement saved with the classifier.
- Split reviewed images by experiment batch, or randomly by original for independent images. Related originals and processed copies stay in one split; existing test splits are retained.
- Export YOLO datasets or reviewed processed images grouped into five class folders.
- Train with CPU, Apple silicon GPU or NVIDIA GPU. Classification runs generate per-class metrics, confusion matrices, prediction CSVs, error lists and training reports.

## Class definitions

| ID | Class | Definition | Shortcut |
|---|---|---|---|
| 0 | `1cell 1bead` | Exactly one cell and one bead | 1 |
| 1 | `only cell` | Cells present, no beads; multiple cells allowed | 2 |
| 2 | `only beads` | Beads present, no cells; multiple beads allowed | 3 |
| 3 | `multiplet` | Both present, with more than one cell or bead | 4 |
| 4 | `empty` | No cells or beads | 5 |

## Getting started

Use Python 3.12 or 3.13. Basic annotation and dataset export use the Python standard library. Training, prediction and enhancement previews need the additional dependencies.

```sh
git clone https://github.com/WenhuaXukiro/microfluidic-droplet-vision-workbench.git
cd microfluidic-droplet-vision-workbench
python3 server.py
```

Open **http://127.0.0.1:8765/** and choose **English** at the top right. Keep the server terminal open. The server binds only to the local loopback address.

### Apple silicon Mac

Install a native ARM Python on macOS 14 or later, then run:

```sh
zsh 安装训练环境.command
zsh 启动软件.command
```

The first script installs dependencies and downloads verified official classification weights. The second starts the workbench. `检查训练环境.command` checks the installed environment and GPU availability.

Alternatively:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_pretrained_weights.py
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python server.py
```

### Windows

Install 64-bit Python 3.12 or 3.13 with the Python Launcher and PATH enabled. Extract or clone the repository into a writable folder, then run:

1. `02_Install_Training_Windows.bat` — install dependencies and classification weights.
2. `01_Start_Windows.bat` — start the workbench.
3. `03_Check_Environment_Windows.bat` — check the environment.

For manual commands, replace `.venv/bin/python` with `.venv\Scripts\python.exe`. The MPS environment variable applies only to Mac. For NVIDIA acceleration, install a compatible CUDA build of PyTorch into this project's `.venv` using the [official PyTorch installer](https://pytorch.org/get-started/locally/).

Dependency installation and initial pretrained weight downloads require internet access. YOLO detection obtains `yolo11n.pt` through Ultralytics when needed.

## Workflow

1. Create a project, assign an experiment batch and import images.
2. For classification, crop complete single-droplet ROIs and generate processed copies. For multi-droplet originals, use object detection annotations.
3. Annotate and review images. Leave unclear images unreviewed.
4. Separate independent experiments or videos into training, validation and test sets. Do not randomly split consecutive frames of the same video. The interface provides batch-based and random-by-original split options.
5. Select the training task, network, device and parameters, then start training. Only reviewed, active images from the selected project are used.
6. Inspect validation metrics and errors, then assess generalization on an independent experiment or video.
7. Click **Predict with model** for a matching image. Accepted predictions are drafts and still require review.

## Training and evaluation

Single-droplet classification preserves the full ROI and its aspect ratio, resizing and padding instead of randomly cropping. The default long edge is 512 pixels. Training uses mild brightness/contrast changes, horizontal and vertical flips, and class-frequency weighting.

The first two epochs adapt the classification head. ConvNeXt-Tiny then freezes the first two feature stages and fine-tunes later stages and the head. The best epoch is selected by validation macro F1, with accuracy as the tie-breaker; early stopping is supported.

A candidate is compared with prior model results only when validation image IDs and labels match exactly. It is activated automatically when both accuracy and macro F1 improve and every class has at least one correct prediction. A majority-class baseline is used when no matching prior results exist. Previous checkpoints are retained.

Classification draws a whole-image box to display its class; it does not learn a droplet boundary. Training console output and generated reports retain their original output language.

### Recorded experiment

On 2026-10-22, a ConvNeXt-Tiny retraining run used 2,150 training images and 537 validation images, holding out later frames of one video. The retrained model correctly classified 522 validation images.

| Metric | Previous model on the same validation set | Retrained model |
|---|---:|---:|
| Accuracy | 93.48% | 97.21% |
| Macro F1 | 86.72% | 95.22% |
| Balanced accuracy | 83.25% | 93.96% |

This validation set was used for checkpoint selection and comes from the same video. These are not independent test results and do not establish performance on new videos or experiments. The experimental images, annotations and weights are not included in this repository.

## Files and local data

| Path | Purpose |
|---|---|
| `server.py` / `worker.py` | Local service and training/prediction workers |
| `static/index.html` / `static/i18n.js` | Workbench UI and bilingual catalog |
| `roi_classifier.py` / `train_roi_classifier.py` | Shared classifier input pipeline, training and validation |
| `image_preprocessing.py` | Consistent enhancement for preview and models |
| `evaluate_model.py` / `evaluate_preprocessing.py` | Dataset auditing and evaluation |
| `download_pretrained_weights.py` | Download and verify official classification weights |
| `data/project.json` | Local project metadata and annotations |
| `data/images/` / `data/exports/` | Local images and dataset snapshots |
| `data/classifier_runs/` / `data/runs/` | Classification and detection runs |
| `models/<project-id>/active_model.json` | Active model references using relative paths |

The GitHub repository contains source code, documentation, installers and checks. `.gitignore` excludes experiment images, annotations, checkpoints, run outputs, complete backup archives and virtual environments. A fresh clone starts with an empty workbench.

To migrate an existing complete project, back it up first, then transfer its matching `data/`, `models/` and `weights/` folders together. Preserve the folder structure. Historical full-package documents describe those backups, not data bundled with the source repository.

## Checks

Basic Python checks without training dependencies:

```sh
python3 -m unittest test_auto_split test_classified test_recycle test_workflow
```

After installing training dependencies:

```sh
.venv/bin/python -m unittest discover -p 'test_*.py'
```

Frontend checks use Node.js:

```sh
node test_preprocess.cjs
node test_quick_label.cjs
node test_trash.cjs
node test_i18n.cjs
```

Node.js is not required to use the workbench or train models.

The workbench classifies droplets or produces detection boxes. It does not output exact cell/bead counts or droplet contours. An independent test-set evaluation screen is not included. PyTorch, TorchVision, Ultralytics and their weights remain subject to their respective licenses.
