def paragraphs(text, min_words=100):
    paras = text.split("\n\n")
    return [p.strip() for p in paras if len(p.split()) >= min_words]
