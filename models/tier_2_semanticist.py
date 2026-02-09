import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel

# ---------- CONFIG ----------
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
encoder.eval()


def embed(text):
    """Average token embeddings"""
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


# Load data
texts, labels = [], []

for path, label in [
    ("data/class1.jsonl", "human"),
    ("data/class2.jsonl", "ai"),
    ("data/class3.jsonl", "ai"),
]:
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(label)

print("Embedding texts...")
X = np.vstack([embed(t) for t in texts])

le = LabelEncoder()
y = le.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Simple Feedforward NN
class Classifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

model = Classifier(X.shape[1]).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
print("Training Semanticist...")
for epoch in range(5):
    optimizer.zero_grad()
    logits = model(torch.tensor(X_train).to(DEVICE))
    loss = criterion(logits, torch.tensor(y_train).to(DEVICE))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    preds = model(torch.tensor(X_test).to(DEVICE)).argmax(dim=1).cpu()

print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=le.classes_))
