# main.py
import json
from babel_folder.utils import epub_to_clean_paragraphs

paras = epub_to_clean_paragraphs("dickens.epub")

with open("data/class1.jsonl", "w") as f:
    for p in paras:
        f.write(json.dumps({
            "text": p,
            "author": "Dickens",
            "topic": "UNKNOWN",
            "class": 1
        }) + "\n")
