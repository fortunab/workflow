import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils_reasoning import format_prompt

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reasoning.yaml")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tokens", default="<DET count='1' avg_conf='0.91' level='high'>\n<SEG masks='1' avg_area='0.0732' />\n<UNCERTAINTY level='low' reason='consistent_detection' />")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model_path or cfg["merged_output_dir"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    model.eval()

    instruction = "Generate a clinical reasoning answer from the structured medical-image tokens."
    prompt = format_prompt(instruction, args.tokens)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(text)

if __name__ == "__main__":
    main()
