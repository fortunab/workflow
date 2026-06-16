from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid

from app.pipeline import run_pipeline

app = FastAPI(title="Workflow-Centric Medical AI API")

UPLOAD_DIR = Path("outputs/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Workflow-Centric Medical AI API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".jpg"
    image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return run_pipeline(str(image_path))

@app.get("/image")
def get_image(path: str):
    return FileResponse(path)
