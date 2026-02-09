# models/tierC_transformer.py
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_lin", "v_lin"])
model = get_peft_model(model, lora)
