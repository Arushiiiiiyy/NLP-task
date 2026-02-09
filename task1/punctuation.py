# task1/punctuation.py
from collections import Counter

PUNCT = [";", "—", "!", ":", ",", "."]

def punctuation_density(text):
    """
    Returns punctuation frequency per 1000 characters.
    """
    counts = Counter(text)
    total = len(text)

    return {
        p: (counts[p] / total) * 1000
        for p in PUNCT
    }
