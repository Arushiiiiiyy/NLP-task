# features/lexical.py
from collections import Counter
import nltk

def ttr(text):
    tokens = nltk.word_tokenize(text.lower())
    return len(set(tokens)) / len(tokens)

def hapax_count(text, n=5000):
    tokens = nltk.word_tokenize(text.lower())[:n]
    freq = Counter(tokens)
    return sum(1 for w in freq if freq[w] == 1)
