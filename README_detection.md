1. Create Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Why?
isolates dependencies
avoids conflicts with Anaconda/base Python
ensures reproducibility

2. Install Dependencies
Python 3.12:
```powershell
pip install -r requirements-py312.txt
```
Why?
installs YOLOv8
installs HuggingFace datasets
installs FastAPI and Shiny
installs scientific libraries

3. Prepare PolypGen Dataset
```powershell
python scripts\prepare_polypgen_yolo.py
```
Why?
downloads PolypGen2.0
creates train/validation split
converts annotations to YOLO format
generates data.yaml
Output:
```text
yolo_dataset/
data.yaml
```

4. Train YOLOv8
```powershell
python scripts\train_yolo.py
```
Why?
This is the Detection Worker (W2) described in the paper.
Output:
```text
runs/detect/polyp_yolov8/weights/best.pt
```

5. Save Best Model
```powershell
mkdir models\yolo
copy runs\detect\polyp_yolov8\weights\best.pt models\yolo\best.pt
```

6. Evaluate Detection
```powershell
python scripts\evaluate_yolo.py
```
Produces:
mAP@0.5
mAP@0.5:0.95
Precision
Recall

7. Run Complete Pipeline
```powershell
python scripts\run_pipeline.py --image examples\sample.jpg
```
Runs:
```text
Image → Detection → Token Generation → Report Generation
```

8. Start API
```powershell
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
Open:
```text
http://localhost:8000/docs
```

9. Start Shiny Interface
```powershell
.venv\Scripts\Activate.ps1
shiny run shiny_app\app.py --reload --port 8001
```
Open:
```text
http://localhost:8001
```
Minimal Reproducibility Workflow
```powershell
python scripts\prepare_polypgen_yolo.py
python scripts\train_yolo.py
copy runs\detect\polyp_yolov8\weights\best.pt models\yolo\best.pt
python scripts\evaluate_yolo.py
python scripts\run_pipeline.py --image examples\sample.jpg
```

### Deactivate venv
```powershell
deactivate
```

