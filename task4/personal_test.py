# task4/personal_test.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/tier_c_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

def check(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)

    return {
        "Human": probs[0, 0].item(),
        "AI": probs[0, 1].item()
    }


if __name__ == "__main__":
    text = open("my_sop.txt").read()
    scores = check(text)
    print(scores)
