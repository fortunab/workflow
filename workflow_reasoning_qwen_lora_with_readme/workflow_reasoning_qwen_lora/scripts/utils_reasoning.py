import json
from pathlib import Path

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def format_prompt(instruction, input_text):
    return (
        "<|system|>\n"
        "You are a cautious medical AI reasoning assistant. "
        "Use the structured tokens only as decision-support evidence. "
        "Do not provide a standalone diagnosis.\n"
        "<|user|>\n"
        f"{instruction}\n\nStructured tokens:\n{input_text}\n"
        "<|assistant|>\n"
    )

def format_training_text(example):
    return format_prompt(example["instruction"], example["input"]) + example["output"]

def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
