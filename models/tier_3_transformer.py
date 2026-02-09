import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# ---------- CONFIG ----------
BASE_MODEL = "distilbert-base-uncased"
OUT_DIR = "models/tier_c"
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

texts, labels = [], []

for path, label in [
    ("data/class1.jsonl", 0),  # human
    ("data/class2.jsonl", 1),  # ai
    ("data/class3.jsonl", 1),  # ai
]:
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(label)

dataset = Dataset.from_dict({"text": texts, "label": labels})

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=2
)

# LoRA
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, config)
model.to(DEVICE)

args = TrainingArguments(
    output_dir=OUT_DIR,
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    
)

trainer.train()
trainer.save_model(OUT_DIR)
