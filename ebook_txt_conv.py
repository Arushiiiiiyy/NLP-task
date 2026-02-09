from ebooklib import epub
from bs4 import BeautifulSoup

def epub_to_text(path):
    book = epub.read_epub(path)
    out = []
    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            out.append(soup.get_text())
    return "\n".join(out)
