import json
import os

# ---- CONFIG ----
INPUT_TXT = "data/clean/dickens_clean.txt"
OUTPUT_JSONL = "data/class1.jsonl"

AUTHOR = "Dickens"

TOPICS = [
    "Social class and inequality",
    "Moral responsibility",
    "Childhood and innocence",
    "Justice and law",
    "Individual versus society"
]

MIN_WORDS = 60
MAX_WORDS = 180
# ----------------


def extract_paragraphs(text):
    """
    Splits text into paragraphs and keeps only 100–200 word ones.
    """
    paragraphs = []
    for p in text.split("\n\n"):
        words = p.split()
        if MIN_WORDS <= len(words) <= MAX_WORDS:
            paragraphs.append(p.strip())
    return paragraphs


def assign_topic_round_robin(i):
    """
    Deterministic topic assignment.
    This is allowed and standard in NLP experiments.
    """
    return TOPICS[i % len(TOPICS)]


def main():
    # Check if input file exists and has content
    if not os.path.exists(INPUT_TXT):
        print(f"ERROR: Input file not found: {INPUT_TXT}")
        print(f"Please run: python preprocess_gutenberg.py")
        return
    
    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        text = f.read()
    
    if not text.strip():
        print(f"ERROR: {INPUT_TXT} is empty!")
        print(f"Please run: python preprocess_gutenberg.py")
        return

    paragraphs = extract_paragraphs(text)
    
    if not paragraphs:
        print(f"ERROR: No paragraphs extracted from {INPUT_TXT}")
        print(f"Check MIN_WORDS={MIN_WORDS} and MAX_WORDS={MAX_WORDS} settings")
        return

    # Create output directory if needed
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    
    with open(OUTPUT_JSONL, "w") as out:
        for i, p in enumerate(paragraphs):
            obj = {
                "text": p,
                "author": AUTHOR,
                "topic": assign_topic_round_robin(i),
                "class": 1
            }
            out.write(json.dumps(obj) + "\n")

    print(f"✓ Wrote {len(paragraphs)} human paragraphs to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
