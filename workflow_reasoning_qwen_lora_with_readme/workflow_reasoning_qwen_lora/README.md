# Qwen LoRA Reasoning Pipeline

This package reproduces the reasoning component of the workflow-centric medical AI pipeline.

```text
Detection Tokens
      ↓
Segmentation Tokens
      ↓
Uncertainty Tokens
      ↓
Qwen + LoRA Reasoning
      ↓
Clinical-style Report
      ↓
ROUGE / BLEU Evaluation
```

## What this component does

This is the reasoning worker:

```text
Worker W4 — Reasoning / VLM
fvlm
```

It receives structured tokens such as:

```text
<DET count='1' avg_conf='0.91' level='high'>
<BOX id='0' label='polyp' conf='0.91' x1='51' y1='42' x2='248' y2='219' />
<SEG masks='1' avg_area='0.0732' />
<UNCERTAINTY level='low' reason='consistent_detection' />
```

and produces a clinical-style reasoning answer.

## Project structure

```text
workflow_reasoning_qwen_lora/
├── README.md
├── requirements.txt
├── configs/
│   └── reasoning.yaml
├── data/
│   ├── train_reasoning.jsonl
│   └── eval_reasoning.jsonl
├── scripts/
│   ├── train_qwen_lora.py
│   ├── merge_lora.py
│   ├── infer_reasoning.py
│   ├── evaluate_reasoning.py
│   ├── build_prompt_from_tokens.py
│   └── utils_reasoning.py
├── models/
│   ├── lora/
│   └── merged/
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

## 3. Check configuration

Open:

```text
configs/reasoning.yaml
```

Default model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

You may replace it with your Qwen model, for example:

```text
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen3-0.6B
```

The uploaded notebook is copied into:

```text
notebooks/
```

## 4. Training data format

Training examples are stored in JSONL format:

```text
data/train_reasoning.jsonl
data/eval_reasoning.jsonl
```

Each row has:

```json
{
  "instruction": "Generate a clinical reasoning answer from the structured medical-image tokens.",
  "input": "<DET ...>\n<SEG ...>\n<UNCERTAINTY ...>",
  "output": "Clinical-style reasoning answer."
}
```

Why this matters:
- the reasoning model learns to convert structured intermediate outputs into explanations
- detection and segmentation outputs become language-model context
- this supports workflow-centric orchestration

## 5. Train Qwen + LoRA

```powershell
python scripts\train_qwen_lora.py
```

Why this matters:
- loads the Qwen model
- adds LoRA adapters
- fine-tunes only a small number of parameters
- saves the adapter to:

```text
models/lora/qwen_reasoning_lora
```

## 6. Merge LoRA adapter

```powershell
python scripts\merge_lora.py
```

Why this matters:
- combines base Qwen with the LoRA adapter
- produces one deployable model
- saves it to:

```text
models/merged/qwen_reasoning_merged
```

## 7. Run inference

```powershell
python scripts\infer_reasoning.py
```

You can also pass your own tokens:

```powershell
python scripts\infer_reasoning.py --tokens "<DET count='1' avg_conf='0.91' level='high'> <SEG masks='1' avg_area='0.0732' /> <UNCERTAINTY level='low' reason='consistent_detection' />"
```

Why this matters:
- tests the reasoning layer
- converts pipeline tokens into a clinical-style answer

## 8. Evaluate reasoning

```powershell
python scripts\evaluate_reasoning.py
```

Reported metrics:
- ROUGE
- BLEU

Output:

```text
outputs/reasoning_metrics.json
```

These metrics correspond to the reasoning/VQA evaluation part of the paper.

## 9. Build prompt from tokens

```powershell
python scripts\build_prompt_from_tokens.py --tokens "<DET count='1' avg_conf='0.91' level='high'> <SEG masks='1' avg_area='0.0732' />"
```

Why this matters:
- shows exactly what is sent to Qwen
- helps debug the orchestration between detection, segmentation, and reasoning

## Minimal reproducibility workflow

```powershell
python scripts\train_qwen_lora.py
python scripts\merge_lora.py
python scripts\infer_reasoning.py
python scripts\evaluate_reasoning.py
```

## Integration with the full pipeline

The complete pipeline is:

```text
Medical Image
      ↓
YOLOv8 Detection
      ↓
SAM2.1 + LoRA Segmentation
      ↓
Structured Tokens
      ↓
Qwen + LoRA Reasoning
      ↓
Clinical Report
```

The reasoning worker receives tokens from previous workers and generates an interpretable answer.

Example final input:

```text
<DET count='1' avg_conf='0.91' level='high'>
<BOX id='0' label='polyp' conf='0.91' x1='51' y1='42' x2='248' y2='219' />
<SEG masks='1' avg_area='0.0732' />
<UNCERTAINTY level='low' reason='consistent_detection' />
```

Example final output:

```text
A polyp-like region is detected with high confidence. The segmentation mask indicates a localized lesion area. The result should be reviewed clinically and is not a standalone diagnosis.
```
