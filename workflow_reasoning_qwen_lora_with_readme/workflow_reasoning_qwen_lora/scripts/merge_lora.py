import argparse
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reasoning.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["lora_output_dir"], trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, cfg["lora_output_dir"])
    merged = model.merge_and_unload()

    out_dir = Path(cfg["merged_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("Merged model saved to:", out_dir)

if __name__ == "__main__":
    main()
