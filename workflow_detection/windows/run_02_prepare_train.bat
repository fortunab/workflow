@echo off
call .venv\Scripts\activate

python scripts\prepare_polypgen_yolo.py
python scripts\train_yolo.py

if not exist models\yolo mkdir models\yolo
copy /Y runs\detect\polyp_yolov8\weights\best.pt models\yolo\best.pt

python scripts\evaluate_yolo.py
pause
