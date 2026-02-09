# task1/lexical.py
from collections import Counter
import re

WORD_RE = re.compile(r"\b\w+\b")

def tokenize(text):
    """
    Lowercases and extracts word tokens.
    """
    return WORD_RE.findall(text.lower())


def type_token_ratio(tokens):
    """
    TTR = unique tokens / total tokens
    """
    return len(set(tokens)) / len(tokens)


def hapax_legomena(tokens):
    """
    Counts tokens that occur exactly once.
    """
    counts = Counter(tokens)
    return sum(1 for c in counts.values() if c == 1)
