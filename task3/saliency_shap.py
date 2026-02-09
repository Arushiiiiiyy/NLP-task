import shap
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, LoraConfig, get_peft_model
import torch
MAX_LEN = 256

with open("data/class2.jsonl") as f:
    IMPOSTER_TEXT = json.loads(next(f))["text"]


BASE_MODEL = "distilbert-base-uncased"
ADAPTER_PATH = "models/tier_c"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2
)

# Load and merge LoRA adapter
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✓ LoRA adapter loaded successfully")
except Exception as e:
    print(f"Warning: Could not load LoRA adapter from {ADAPTER_PATH}: {e}")
    print("Using base model without LoRA")
    model = base_model

model.to(DEVICE)
model.eval()



def predict(texts):

    cleaned_texts = []

    for t in texts:
        if isinstance(t, (list, np.ndarray)):
       
            t = " ".join([x for x in t if x is not None])
        elif t is None:
            t = ""

        cleaned_texts.append(str(t))

    inputs = tokenizer(
        cleaned_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)


    return probs[:, 1].cpu().numpy()



explainer = shap.Explainer(predict, tokenizer)


shap_values = explainer([IMPOSTER_TEXT])


shap.plots.text(shap_values[0])
