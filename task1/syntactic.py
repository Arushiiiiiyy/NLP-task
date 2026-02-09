# task1/syntactic.py
import spacy

nlp = spacy.load("en_core_web_sm")

def parse(text):
    """
    Runs SpaCy once and returns a Doc.
    """
    return nlp(text)


def adjective_noun_ratio(doc):
    """
    Ratio of adjectives to nouns.
    """
    adj = sum(1 for t in doc if t.pos_ == "ADJ")
    noun = sum(1 for t in doc if t.pos_ == "NOUN")
    return adj / noun if noun > 0 else 0.0


def sentence_tree_depth(sent):
    """
    Computes maximum dependency depth of a sentence.
    """
    def depth(token):
        if not list(token.children):
            return 1
        return 1 + max(depth(child) for child in token.children)

    return depth(sent.root)


def average_dependency_depth(doc):
    """
    Average dependency tree depth across sentences.
    """
    depths = [sentence_tree_depth(sent) for sent in doc.sents]
    return sum(depths) / len(depths)
