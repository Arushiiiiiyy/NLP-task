import os
import sys
import json
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from task1.lexical import tokenize, type_token_ratio, hapax_legomena
from task1.syntactic import parse, adjective_noun_ratio, average_dependency_depth
from task1.punctuation import punctuation_density
from task1.readability import flesch_kincaid
DATASETS = {
    1: "data/class1.jsonl",
    2: "data/class2.jsonl",
    3: "data/class3.jsonl"
}

def extract_features(text):
    tokens = tokenize(text)
    doc = parse(text)
    punct = punctuation_density(text)

    return {
        "ttr": type_token_ratio(tokens),
        "hapax": hapax_legomena(tokens),
        "adj_noun_ratio": adjective_noun_ratio(doc),
        "dep_depth": average_dependency_depth(doc),
        "fk_grade": flesch_kincaid(text),
        **punct
    }

def run():
    rows = []

    for label, path in DATASETS.items():
        with open(os.path.join(BASE_DIR, path)) as f:
            for line in f:
                obj = json.loads(line)
                feats = extract_features(obj["text"])
                feats["label"] = label
                rows.append(feats)

    out = os.path.join(BASE_DIR, "features/task1_features.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    run()
