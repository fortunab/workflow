# Reproducibility and Extension Instructions for the Article

To reproduce the detection-oriented part of the workflow, the reader should execute the following steps.

## Step 1. Create an isolated environment

Use Python 3.10 where possible. Python 3.12 can also be used with the alternative requirements file.

## Step 2. Prepare the dataset

Run:

```bash
python scripts/prepare_polypgen_yolo.py
```

This downloads PolypGen2.0 from Hugging Face, splits it into train/validation subsets, and converts bounding boxes to YOLO format.

## Step 3. Train YOLOv8

Run:

```bash
python scripts/train_yolo.py
```

This fine-tunes YOLOv8 on the generated dataset.

## Step 4. Save the best model

Copy:

```text
runs/detect/polyp_yolov8/weights/best.pt
```

to:

```text
models/yolo/best.pt
```

## Step 5. Evaluate detection

Run:

```bash
python scripts/evaluate_yolo.py
```

Report mAP@0.5, mAP@0.5:0.95, precision, and recall.

## Step 6. Run the full modular pipeline

Run:

```bash
python scripts/run_pipeline.py --image path/to/image.jpg
```

The script returns detections, structured tokens, and a clinical-style report.

## Step 7. Launch the API

Run:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## Step 8. Launch the Shiny/Posit interface

Run:

```bash
shiny run shiny_app/app.py --reload --port 8001
```

## How to continue the work

The current implementation is intentionally modular. Future work may replace:

- YOLOv8 with YOLOv9, YOLOv10, RT-DETR, or another detector.
- The fallback segmentation function with SAM2.1 + LoRA or MedSAM.
- The template report generator with a fine-tuned Qwen/VLM model.
- The local pipeline with a federated deployment strategy.
