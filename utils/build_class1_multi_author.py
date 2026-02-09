"""
Build Class 1 dataset from multiple authors.
Extracts paragraphs from both authors' novels with extracted topics.
"""

import json
import os

# CONFIG
AUTHORS_CONFIG = {
    "Dickens": {
        "clean_file": "data/clean/dickens_clean.txt",
        "author_name": "Charles Dickens"
    },
    "Austen": {
        "clean_file": "data/clean/austen_clean.txt",
        "author_name": "Jane Austen"
    }
}

OUTPUT_JSONL = "data/class1.jsonl"

MIN_WORDS = 60
MAX_WORDS = 180

# Load topics from config
def load_topics():
    """Load topics from the generated config file."""
    if os.path.exists("data/topics_config.json"):
        with open("data/topics_config.json", "r") as f:
            config = json.load(f)
            return config
    return None

def extract_paragraphs(text, min_words=60, max_words=180):
    """
    Splits text into paragraphs and keeps only those within word count range.
    """
    paragraphs = []
    for p in text.split("\n\n"):
        words = p.split()
        if min_words <= len(words) <= max_words:
            paragraphs.append(p.strip())
    return paragraphs

def assign_topic_round_robin(i, topics):
    """
    Deterministic topic assignment.
    """
    return topics[i % len(topics)]

def main():
    print("=" * 70)
    print("Building Class 1 Dataset (Human-Written Text)")
    print("=" * 70)
    
    # Load topics
    topics_config = load_topics()
    if not topics_config:
        print("ERROR: Could not load topics_config.json")
        print("Please run: python preprocess_multi_author.py")
        return
    
    all_paragraphs = []
    
    # Process each author
    for short_name, author_info in AUTHORS_CONFIG.items():
        clean_file = author_info["clean_file"]
        author_name = author_info["author_name"]
        topics = topics_config[short_name]["topics"]
        
        print(f"\n{author_name}:")
        
        if not os.path.exists(clean_file):
            print(f"  ✗ File not found: {clean_file}")
            print(f"  Please run: python preprocess_multi_author.py")
            continue
        
        with open(clean_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        if not text.strip():
            print(f"  ✗ File is empty: {clean_file}")
            continue
        
        paragraphs = extract_paragraphs(text, MIN_WORDS, MAX_WORDS)
        
        if not paragraphs:
            print(f"  ✗ No paragraphs extracted (check word count settings)")
            continue
        
        print(f"  ✓ Extracted {len(paragraphs)} paragraphs")
        print(f"  Topics: {', '.join(topics)}")
        
        # Create paragraph objects
        for i, p in enumerate(paragraphs):
            obj = {
                "text": p,
                "author": author_name,
                "topic": assign_topic_round_robin(i, topics),
                "class": 1,
                "source": "human"
            }
            all_paragraphs.append(obj)
    
    if not all_paragraphs:
        print("\nERROR: No paragraphs were extracted from any author!")
        return
    
    # Write to JSONL
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    
    with open(OUTPUT_JSONL, "w") as out:
        for obj in all_paragraphs:
            out.write(json.dumps(obj) + "\n")
    
    print(f"\n{'=' * 70}")
    print(f"✓ SUCCESS! Wrote {len(all_paragraphs)} human paragraphs to:")
    print(f"  {OUTPUT_JSONL}")
    print(f"\nDataset breakdown:")
    
    for short_name in AUTHORS_CONFIG.keys():
        count = sum(1 for p in all_paragraphs if p["author"] == AUTHORS_CONFIG[short_name]["author_name"])
        print(f"  - {short_name}: {count} paragraphs")
    
    print(f"\n{'=' * 70}")
    print("Next step: Generate AI paragraphs")
    print("  python generate_gemini.py")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
