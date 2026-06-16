import argparse
import yaml
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils_reasoning import read_jsonl, format_prompt, save_json

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_answer(model, tokenizer, prompt, max_new_tokens=160):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "<|assistant|>" in generated:
        generated = generated.split("<|assistant|>")[-1].strip()
    return generated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reasoning.yaml")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--out", default="outputs/reasoning_metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model_path or cfg["merged_output_dir"]

    rows = read_jsonl(cfg["eval_jsonl"])

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    model.eval()

    predictions, references = [], []

    for row in rows:
        prompt = format_prompt(row["instruction"], row["input"])
        pred = generate_answer(model, tokenizer, prompt)
        predictions.append(pred)
        references.append(row["output"])

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_scores = bleu.compute(predictions=predictions, references=[[r] for r in references])

    metrics = {
        "rouge": rouge_scores,
        "bleu": bleu_scores,
        "predictions": predictions,
        "references": references
    }

    save_json(args.out, metrics)

    print("ROUGE:", rouge_scores)
    print("BLEU:", bleu_scores["score"])
    print("Saved metrics to:", args.out)

if __name__ == "__main__":
    main()
