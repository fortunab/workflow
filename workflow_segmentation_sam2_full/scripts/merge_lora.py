import argparse
from pathlib import Path
import yaml
from transformers import Sam2Model, Sam2Processor
from peft import PeftModel

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    base_model = Sam2Model.from_pretrained(cfg["model_id"])
    model = PeftModel.from_pretrained(base_model, cfg["lora_output_dir"])

    merged_model = model.merge_and_unload()

    out_dir = Path(cfg["merged_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(out_dir)

    processor = Sam2Processor.from_pretrained(cfg["model_id"])
    processor.save_pretrained(out_dir)

    print("Merged model saved to:", out_dir)

if __name__ == "__main__":
    main()
