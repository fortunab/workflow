from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import uuid

from scripts.orchestrate_pipeline import run_orchestration

app = FastAPI(title="Workflow-Centric Orchestrator API")

UPLOAD_DIR = Path("outputs/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Workflow orchestrator is running."}

@app.post("/orchestrate")
async def orchestrate(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".jpg"
    image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return run_orchestration(str(image_path))

@app.get("/demo")
def demo():
    return run_orchestration(use_demo=True)
