# task1/load_data.py
import json

def load_class(path):
    """
    Loads all text entries from a JSONL file.
    """
    texts = []
    with open(path, "r") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def sample_5000_words(texts, limit=5000):
    """
    Concatenates text until `limit` words are reached.
    Deterministic (important for reproducibility).
    """
    words = []
    for t in texts:
        words.extend(t.split())
        if len(words) >= limit:
            break
    return " ".join(words[:limit])
