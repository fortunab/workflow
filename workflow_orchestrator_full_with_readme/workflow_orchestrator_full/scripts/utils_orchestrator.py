from pathlib import Path
import yaml
import json

def load_config(path="configs/orchestrator.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def confidence_level(score, low=0.45, high=0.75):
    if score >= high:
        return "high"
    if score >= low:
        return "medium"
    return "low"
