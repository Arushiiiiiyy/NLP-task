# task1/run_task1.py

from load_data import load_class, sample_5000_words
from lexical import tokenize, type_token_ratio, hapax_legomena
from syntactic import parse, adjective_noun_ratio, average_dependency_depth
from punctuation import punctuation_density
from readability import flesch_kincaid
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = {
    "Human (Dickens)": os.path.join(BASE_DIR,"data","class1.jsonl"),
    "Gemini-Neutral": os.path.join(BASE_DIR, "data", "class2.jsonl"),
    "Gemini-Dickens": os.path.join(BASE_DIR, "data", "class3.jsonl")
}

def analyze(label, path):
    print(f"\n===== {label} =====")

    texts = load_class(path)
    sample = sample_5000_words(texts)

    tokens = tokenize(sample)
    doc = parse(sample)

    print(f"TTR: {type_token_ratio(tokens):.4f}")
    print(f"Hapax Legomena: {hapax_legomena(tokens)}")
    print(f"Adj/Noun Ratio: {adjective_noun_ratio(doc):.4f}")
    print(f"Avg Dependency Depth: {average_dependency_depth(doc):.4f}")
    print(f"Punctuation Density: {punctuation_density(sample)}")
    print(f"Flesch–Kincaid Grade: {flesch_kincaid(sample):.2f}")


if __name__ == "__main__":
    for label, path in CLASSES.items():
        analyze(label, path)
