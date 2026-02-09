import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# ---------- CONFIG ----------
BASE_MODEL = "distilbert-base-uncased"
ADAPTER_PATH = "models/tier_c"
DATA_PATH = "data/class1.jsonl"   # HUMAN ONLY
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model and LoRA adapter
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2
)

try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✓ LoRA adapter loaded successfully")
except Exception as e:
    print(f"Warning: Could not load LoRA adapter: {e}")
    model = base_model

model.to(DEVICE)
model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    return probs[0, 1].item()  # AI probability

# Load human samples
misclassified = []
total_samples = 0

print("\nAnalyzing human text classifications...")

with open(DATA_PATH) as f:
    for idx, line in enumerate(f):
        obj = json.loads(line)
        text = obj["text"]
        ai_prob = predict(text)
        total_samples += 1

        if ai_prob > 0.7:  # confident AI prediction
            misclassified.append((ai_prob, text, obj.get("author", "Unknown")))
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1} samples...")

# Sort by confidence
misclassified.sort(reverse=True, key=lambda x: x[0])

print(f"\nResults:")
print(f"  Total human samples: {total_samples}")
print(f"  Misclassified as AI (>70%): {len(misclassified)}")

if len(misclassified) > 0:
    print(f"\nTop 3 HUMAN texts misclassified as AI:\n")
    for i, (score, text, author) in enumerate(misclassified[:3], 1):
        print(f"\n--- Sample {i} (AI prob = {score:.3f}, Author: {author}) ---")
        print(text[:500])
        print(f"[... truncated, total length: {len(text)} chars]")
else:
    print(f"\n✓ No human texts misclassified as AI!")
    print(f"Model correctly identified all {total_samples} human samples.")
    
    # Show score distribution instead
    print(f"\nScore distribution for human texts:")
    scores = []
    with open(DATA_PATH) as f:
        for line in f:
            obj = json.loads(line)
            scores.append(predict(obj["text"]))
    
    import statistics
    print(f"  Min AI score: {min(scores):.4f}")
    print(f"  Max AI score: {max(scores):.4f}")
    print(f"  Mean AI score: {statistics.mean(scores):.4f}")
    print(f"  Median AI score: {statistics.median(scores):.4f}")