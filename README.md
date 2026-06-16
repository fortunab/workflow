# Workflow

This repository contains Python experiment scripts.

You can run either one specific experiment or all experiments together using `src/run\_all.py`.


## 1\. Install Git

### Windows

Download and install Git for Windows:

https://git-scm.com/download/win

After installation, open a new terminal and verify Git:

```powershell
git --version
```

Expected output:

```text
git version 2.xx.x.windows.x
```

### Linux/macOS

Git is often already installed. Check with:

```bash
git --version
```

If Git is missing, install it using your system package manager.

\---

## 2\. Clone the repository

```bash
git clone https://github.com/fortunab/workflow.git
cd workflow
```

\---

## 3\. Create a virtual environment

### Windows

```powershell
python -m venv .venv
```

### Linux/macOS

```bash
python3 -m venv .venv
```

\---

## 4\. Install dependencies

### Windows

```powershell
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

### Linux/macOS

```bash
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

\---

# Running the experiments

## Option A: Run everything

Use this command to execute all available experiment scripts.

### Windows

```powershell
.\\.venv\\Scripts\\python.exe src\\run\_all.py
```

### Linux/macOS

```bash
./.venv/bin/python src/run\_all.py
```

\---

## Option B: Run scripts independently

You can run each experiment separately.

### Windows

```powershell
.\\.venv\\Scripts\\python.exe src\\segmentation\_metrics.py
.\\.venv\\Scripts\\python.exe src\\inference\_efficiency.py
.\\.venv\\Scripts\\python.exe src\\vqa\_evaluation\_small.py
.\\.venv\\Scripts\\python.exe src\\vqa\_evaluation\_large.py
.\\.venv\\Scripts\\python.exe src\\rougel\_comparison.py
.\\.venv\\Scripts\\python.exe src\\bleu\_comparison.py
.\\.venv\\Scripts\\python.exe src\\pipeline\_metrics.py
.\\.venv\\Scripts\\python.exe src\\latency\_efficiency.py
.\\.venv\\Scripts\\python.exe src\\token\_ablation.py
.\\.venv\\Scripts\\python.exe src\\perception\_analysis.py
.\\.venv\\Scripts\\python.exe src\\transfer\_analysis.py
.\\.venv\\Scripts\\python.exe src\\cross\_dataset\_transfer.py
.\\.venv\\Scripts\\python.exe src\\pipeline\_stage\_latency.py
.\\.venv\\Scripts\\python.exe src\\domain\_token\_ablation.py
.\\.venv\\Scripts\\python.exe src\\uncertainty\_behavior.py
.\\.venv\\Scripts\\python.exe src\\segmentation\_ablation.py
```

### Linux/macOS

```bash
./.venv/bin/python src/segmentation\_metrics.py
./.venv/bin/python src/inference\_efficiency.py
./.venv/bin/python src/vqa\_evaluation\_small.py
./.venv/bin/python src/vqa\_evaluation\_large.py
./.venv/bin/python src/rougel\_comparison.py
./.venv/bin/python src/bleu\_comparison.py
./.venv/bin/python src/pipeline\_metrics.py
./.venv/bin/python src/latency\_efficiency.py
./.venv/bin/python src/token\_ablation.py
./.venv/bin/python src/perception\_analysis.py
./.venv/bin/python src/transfer\_analysis.py
./.venv/bin/python src/cross\_dataset\_transfer.py
./.venv/bin/python src/pipeline\_stage\_latency.py
./.venv/bin/python src/domain\_token\_ablation.py
./.venv/bin/python src/uncertainty\_behavior.py
./.venv/bin/python src/segmentation\_ablation.py
```

\---

## Optional: activate the virtual environment

Instead of writing the full `.venv` Python path every time, you can activate the virtual environment.

### Windows CMD

```cmd
.venv\\Scripts\\activate.bat
```

### Windows PowerShell

```powershell
.venv\\Scripts\\Activate.ps1
```

If PowerShell blocks script execution, use the non-activation commands above instead.

### Linux/macOS

```bash
source .venv/bin/activate
```

After activation, you can run scripts in the following way:

```bash
python src/segmentation\_metrics.py
python src/run\_all.py
```

\---

## Output

Generated results and figures are saved in folders such as:

```text
results/
figures/
```



