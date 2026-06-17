# Workflow GradCAM

Strict GradCAM module for the `fortunab/workflow` project.

This repository contains a structured Python version of the notebook:

```text
GradCAM_Elsevier_article.ipynb
```

It trains an Alzheimer MRI CNN and generates visual explanations using:

```text
GradCAM
GradCAM++
ScoreCAM
```

---

## Repository structure

```text
workflow_gradcam/
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── CITATION.cff
├── configs/
│   └── default.yaml
├── src/
│   └── workflow_gradcam/
│       ├── __init__.py
│       ├── cam.py
│       ├── data.py
│       ├── model.py
│       └── utils.py
├── scripts/
│   ├── train_model.py
│   ├── generate_cams.py
│   └── run_full_pipeline.py
    └── alzheimer_cnn_cam_pipeline.py
├── notebooks/
│   └── GradCAM_Elsevier_article.ipynb
└── outputs/
    ├── figures/
    ├── models/
    └── results/
```

---

## Required Python version

Recommended:

```text
Python 3.10
```

Why?

TensorFlow and `tf-keras-vis` are version-sensitive. Python 3.10 is a safer choice than the newest Python version for reproducible GradCAM experiments.

---

## Option A: venv

### 1. Create a virtual environment

```bash
python -m venv .venv
```

This creates an isolated environment for this GradCAM module.

### 2. Activate the virtual environment

Windows PowerShell:

```bash
.venv/Scripts/Activate.ps1
```

Windows CMD:

```bash
.venv/Scripts/activate.bat
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install packages

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Run the required commands

Train the CNN:

```bash
python scripts/train_model.py
```

Generate GradCAM, GradCAM++, and ScoreCAM for one random test image:

```bash
python scripts/generate_cams.py --mode single
```

Generate CAMs for all classes on one image:

```bash
python scripts/generate_cams.py --mode all-classes --output outputs/figures/all_classes_gradcam.png
```

Generate one example per class:

```bash
python scripts/generate_cams.py --mode one-per-class --output outputs/figures/one_per_class_gradcam.png
```

Run the full pipeline:

```bash
python scripts/run_full_pipeline.py
```

### 5. Deactivate the environment

```bash
deactivate
```

---

## Option B: Conda

### 1. Create environment

```bash
conda create -n workflow-gradcam python=3.10 -y
```

or:

```bash
conda env create -f environment.yml
```

### 2. Activate environment

```bash
conda activate workflow-gradcam
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Run commands

```bash
python scripts/train_model.py
python scripts/generate_cams.py --mode single
```

### 5. Deactivate

```bash
conda deactivate
```

---

## Dataset

The code uses:

```text
yasserhessein/dataset-alzheimer
```

through `kagglehub`.

Expected dataset layout:

```text
Alzheimer_s Dataset/
├── train/
│   ├── MildDemented/
│   ├── ModerateDemented/
│   ├── NonDemented/
│   └── VeryMildDemented/
└── test/
    ├── MildDemented/
    ├── ModerateDemented/
    ├── NonDemented/
    └── VeryMildDemented/
```

The dataset is downloaded automatically when `--dataset-path` is not provided.

To use a local dataset:

```bash
python scripts/train_model.py --dataset-path path/to/Alzheimer_s Dataset
python scripts/generate_cams.py --dataset-path path/to/Alzheimer_s Dataset
```

---

## Outputs

Training saves:

```text
outputs/models/best_model.keras
outputs/results/train_metrics.json
```

CAM generation saves:

```text
outputs/figures/gradcam_result.png
outputs/results/gradcam_metadata.json
```

---

## Minimal reproducibility workflow

Windows PowerShell:

```bash
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python scripts/train_model.py
python scripts/generate_cams.py --mode single
deactivate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python scripts/train_model.py
python scripts/generate_cams.py --mode single
deactivate
```

---

## Notes for `fortunab/workflow`

This module can be placed under the workflow repository as:

```text
workflow/modules/gradcam/
```

The main reusable functions are:

```python
from workflow_gradcam.cam import plot_single_image_cams
from workflow_gradcam.cam import plot_all_class_cams_for_one_image
from workflow_gradcam.cam import plot_one_image_per_class
```

The generated figure can then be consumed by the larger workflow pipeline.
