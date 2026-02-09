# utils.py
import re
from ebooklib import epub
from bs4 import BeautifulSoup

def epub_to_clean_paragraphs(path, min_words=100):
    book = epub.read_epub(path)
    text = []

    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text.append(soup.get_text())

    text = "\n".join(text)

    # remove chapter titles
    text = re.sub(r"\n\s*CHAPTER.*\n", "\n", text)

    paras = [p.strip() for p in text.split("\n\n")]
    return [p for p in paras if len(p.split()) >= min_words]
