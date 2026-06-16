# SAM2.1 + LoRA Segmentation Pipeline

This ZIP contains the full segmentation component:

```text
Medical image + bounding box prompt
      ↓
SAM2.1-hiera-tiny
      ↓
LoRA fine-tuning
      ↓
Merged model
      ↓
Mask prediction
      ↓
IoU / Dice evaluation
```

## 1. Create and activate venv

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

or run commands through:

```powershell
.\.venv\Scripts\python.exe
```

## 2. Install dependencies

```powershell
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 3. Train SAM2.1 + LoRA

```powershell
python scripts\train_sam_lora.py
```

Why this matters:
- loads `RGarrido03/kvasir-seg-augmented`
- loads `facebook/sam2.1-hiera-tiny`
- adds LoRA adapters
- trains segmentation using BCE + Dice loss
- saves adapter to `models/lora/sam2.1-kvasir-lora-final`

## 4. Merge LoRA adapter

```powershell
python scripts\merge_lora.py
```

Why this matters:
- merges LoRA into the base SAM2.1 model
- creates one deployable model
- saves it to `models/merged/sam2_kvasir_merged`

## 5. Evaluate segmentation

```powershell
python scripts\evaluate_sam.py
```

Why this matters:
- evaluates on the test split
- reports mIoU and Dice
- these are the segmentation metrics used in the paper

## 6. Compare base SAM2.1 vs LoRA SAM2.1

```powershell
python scripts\compare_base_vs_lora.py
```

Why this matters:
- checks whether LoRA improves over the base model
- supports the ablation section

## 7. Inference on one image

You need a bounding box prompt:

```powershell
python scripts\infer_sam.py --image examples\sample.jpg --box 50,40,250,220
```

Why this matters:
- SAM needs a prompt
- in the complete pipeline, this box comes from YOLOv8
- output is saved to `outputs/inference_overlay.png`

## Minimal run order

```powershell
python scripts\train_sam_lora.py
python scripts\merge_lora.py
python scripts\evaluate_sam.py
python scripts\compare_base_vs_lora.py
python scripts\infer_sam.py --image examples\sample.jpg --box 50,40,250,220
```

## Files

```text
scripts/train_sam_lora.py
scripts/merge_lora.py
scripts/evaluate_sam.py
scripts/compare_base_vs_lora.py
scripts/infer_sam.py
scripts/utils_segmentation.py
configs/segmentation.yaml
requirements.txt
notebooks/
```

## Mapping to the paper

This corresponds to:

```text
Worker W3 — Segmentation
fseg
```

It receives:

```text
image + bounding box
```

and returns:

```text
binary mask + mask area + segmentation token
```

Example token for the full orchestration pipeline:

```text
<SEG masks='1' avg_area='0.0732' />
```
