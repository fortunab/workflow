import argparse
import yaml
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from utils_reasoning import read_jsonl, format_training_text

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reasoning.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    rows = read_jsonl(cfg["train_jsonl"])
    texts = [{"text": format_training_text(row)} for row in rows]
    dataset = Dataset.from_list(texts)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=int(cfg["max_seq_length"])
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=cfg["lora_output_dir"],
        num_train_epochs=int(cfg["epochs"]),
        per_device_train_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator
    )

    trainer.train()

    model.save_pretrained(cfg["lora_output_dir"])
    tokenizer.save_pretrained(cfg["lora_output_dir"])
    print("LoRA adapter saved to:", cfg["lora_output_dir"])

if __name__ == "__main__":
    main()
