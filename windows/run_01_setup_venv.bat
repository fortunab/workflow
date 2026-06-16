@echo off
echo Creating venv with current Python...
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel

echo Installing Python 3.12-compatible requirements...
pip install -r requirements-py312.txt

echo Done.
python -c "import numpy, pandas, datasets, torch, cv2; print('Environment OK')"
pause
