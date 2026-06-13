# Structured Multimodal Medical Project

This folder reorganizes the uploaded notebooks into a cleaner Python project.

## Source notebooks
- YeaOrchestrators.ipynb
- YOLO_tuning_2bboxs (1).ipynb
- SAM2_1_Tuning_3segmentation (1).ipynb
- Qwen3_5_0_8B_LoRA_vlms_4reasoning (1).ipynb
- orchestrare_R_ULIx (1).ipynb

## Structure
- `project/configs/`: shared settings
- `project/models/`: reusable model classes
- `project/pipelines/`: orchestration logic
- `project/training/`: YOLO, SAM2, and Qwen logic
- `project/evaluation/`: reporting helpers
- `project/utils/`: text/json writers
- `project/scripts/`: runnable demos
- `outputs/`: generated outputs

- # Project 

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run everything

From the repository root:

```bash
python src/run_all.py
```

This generates all CSV files in `results/` and all plots in `figures/`.


## Notes
- Notebook duplicates and exploratory cells were consolidated into reusable modules.
- Heavy training code was converted into reusable stubs/helpers, while demo scripts generate lightweight outputs.
