# features/syntactic.py
import spacy
nlp = spacy.load("en_core_web_sm")

def pos_ratio(text):
    doc = nlp(text)
    adj = sum(1 for t in doc if t.pos_ == "ADJ")
    noun = sum(1 for t in doc if t.pos_ == "NOUN")
    return adj / (noun + 1)

def avg_dependency_depth(text):
    doc = nlp(text)
    depths = []
    for sent in doc.sents:
        for token in sent:
            d = 0
            head = token
            while head.head != head:
                d += 1
                head = head.head
            depths.append(d)
    return sum(depths) / len(depths)
