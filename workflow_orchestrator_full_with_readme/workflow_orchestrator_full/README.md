# Workflow-Centric Orchestrator

This package reproduces the orchestration layer of the medical AI workflow.

```text
Medical Image
      ↓
Detection Worker
      ↓
Segmentation Worker
      ↓
Token Builder
      ↓
Reasoning Worker
      ↓
Clinical-style Report
```

## What this package does

The orchestrator connects separate model workers into one sequential workflow.

It can receive outputs from:

```text
YOLOv8 Detection
SAM2.1 + LoRA Segmentation
Qwen + LoRA Reasoning
```

and convert them into structured tokens:

```text
<DET count='1' avg_conf='0.91' level='high'>
<BOX id='0' label='polyp' conf='0.91' x1='51' y1='42' x2='248' y2='219' />
<SEG masks='1' avg_area='0.0732' />
<UNCERTAINTY level='low' reason='consistent_detection' />
```

## Project structure

```text
workflow_orchestrator_full/
├── README.md
├── requirements.txt
├── configs/
│   └── orchestrator.yaml
├── scripts/
│   ├── orchestrate_pipeline.py
│   ├── token_builder.py
│   ├── reasoning_template.py
│   ├── run_demo.py
│   └── utils_orchestrator.py
├── app/
│   └── api.py
├── examples/
│   └── tokens_example.txt
├── outputs/
└── notebooks/
```

## 1. Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 2. Install dependencies

```powershell
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 3. Run demo orchestration

```powershell
python scripts\run_demo.py
```

Why this matters:
- tests the orchestration logic without external APIs
- creates detection, segmentation, uncertainty, and reasoning outputs
- verifies token generation

## 4. Build tokens only

```powershell
python scripts\token_builder.py --demo
```

Expected output:

```text
<DET count='1' avg_conf='0.910' level='high'>
<BOX id='0' label='polyp' conf='0.910' x1='51.0' y1='42.0' x2='248.0' y2='219.0' />
<SEG masks='1' avg_area='0.0732' />
<UNCERTAINTY level='low' reason='consistent_detection' />
```

## 5. Run orchestrator with demo data

```powershell
python scripts\orchestrate_pipeline.py --demo
```

Output:

```text
outputs/orchestration_result.json
```

## 6. Run orchestrator with detection API

First start the YOLO detection API from the detection package:

```powershell
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Then run:

```powershell
python scripts\orchestrate_pipeline.py --image examples\sample.jpg
```

The orchestrator calls:

```text
http://localhost:8000/predict
```

as configured in:

```text
configs/orchestrator.yaml
```

## 7. Start orchestrator API

```powershell
uvicorn app.api:app --host 0.0.0.0 --port 8010
```

Open:

```text
http://localhost:8010/docs
```

Demo endpoint:

```text
http://localhost:8010/demo
```

## Important logic

The orchestrator supports selective execution.

If detection confidence is too low:

```text
skip segmentation
return uncertainty-aware report
```

If detection confidence is high:

```text
run segmentation
build segmentation token
send tokens to reasoning
```

This implements the workflow-centric idea:

```text
classification/detection output guides segmentation
segmentation output guides reasoning
reasoning output becomes the report
```

## Minimal reproducibility workflow

```powershell
python scripts\token_builder.py --demo
python scripts\orchestrate_pipeline.py --demo
uvicorn app.api:app --host 0.0.0.0 --port 8010
```

## Mapping to the paper

This corresponds to the orchestration layer:

```text
Sequential execution
State transitions
Semantic propagation
Token-guided selective execution
```

The orchestrator is responsible for connecting:

```text
W1 — Classification
W2 — Detection
W3 — Segmentation
W4 — Reasoning / VLM
```

into a unified workflow.
