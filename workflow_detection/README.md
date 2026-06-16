# Workflow-Centric Medical AI Reproducibility Package v2

This package reproduces the practical pipeline:

`medical image -> YOLOv8 detection -> segmentation placeholder/SAM extension -> structured tokens -> VLM/template reasoning -> report -> FastAPI -> Shiny/Posit UI`

Important: the training-heavy components are implemented in Python. Shiny/Posit is used as the visual interface.

## What is included

```text
scripts/prepare_polypgen_yolo.py     Prepare PolypGen2.0 in YOLO format
scripts/train_yolo.py                Train YOLOv8
scripts/evaluate_yolo.py             Evaluate YOLOv8
scripts/run_pipeline.py              Run one image through the pipeline
app/pipeline.py                      Orchestration logic
app/api.py                           FastAPI backend
shiny_app/app.py                     Shiny/Posit interface
configs/pipeline.yaml                Pipeline configuration
requirements-py310.txt               Recommended dependencies for Python 3.10
requirements-py312.txt               Alternative dependencies for Python 3.12
windows/run_01_setup_venv.bat        Windows setup helper
windows/run_02_prepare_train.bat     Windows prepare/train/evaluate helper
windows/run_03_api.bat               Windows FastAPI helper
windows/run_04_shiny.bat             Windows Shiny helper
```

## Recommended environment

Use Python 3.10 for reproducibility.

If you only have Python 3.12, use `requirements-py312.txt`, but Python 3.10 is still recommended for AI reproducibility.

## Pipeline

1. Prepare YOLO dataset from Hugging Face PolypGen2.0.
2. Train YOLOv8.
3. Copy trained weights to `models/yolo/best.pt`.
4. Evaluate YOLOv8.
5. Run the inference pipeline.
6. Start FastAPI.
7. Start Shiny/Posit interface.

## Notes

The included segmentation component is a bounding-box mask fallback so that the full pipeline runs immediately.
For article-level experiments, replace `fallback_bbox_segmentation()` in `app/pipeline.py` with SAM2.1 + LoRA inference.
