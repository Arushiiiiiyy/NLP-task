"""
Multi-author preprocessing and topic extraction.
Cleans texts from multiple authors and identifies core topics.
"""

import os
import re
import json
from collections import Counter

def clean_gutenberg_text(raw_text):
    """
    Removes Project Gutenberg headers, footers, and formatting artifacts.
    """
    
    text = re.sub(
        r"^\*{3}.*?\*{3}",
        "",
        raw_text,
        flags=re.DOTALL | re.MULTILINE
    )
    
   
    text = re.sub(
        r"\*{3}.*?$",
        "",
        text,
        flags=re.DOTALL | re.MULTILINE
    )
    

    text = re.sub(r"\n\s*(CHAPTER|Chapter|PART|Part|BOOK|Book)\s+[A-Z0-9IVX\.\-]+.*?\n", "\n", text)
    

    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    
    
    text = re.sub(r"\[Page \d+\]", "", text)
    text = re.sub(r"\[Illustration:.*?\]", "", text)
    
    return text.strip()


def extract_topics_from_text(text, author_name, num_topics=7):
    """
    Extract core topics by analyzing word frequencies and contextual clues.
    """
    # Common stopwords to exclude
    stopwords = set([
        'the', 'and', 'a', 'an', 'is', 'are', 'was', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'that', 'this',
        'it', 'or', 'of', 'to', 'in', 'on', 'at', 'for', 'with', 'by',
        'from', 'up', 'about', 'into', 'through', 'during', 'as', 'if',
        'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all',
        'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'mr', 'mrs', 'miss', 'said', 'say', 'said',
        'would', 'could', 'one', 'two', 'three', 'first', 'second', 'time',
        'man', 'woman', 'people', 'day', 'year', 'hand', 'eye', 'face'
    ])
    

    words = re.findall(r'\b[a-z]+\b', text.lower())
    

    filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
    word_freq = Counter(filtered_words)
    

    topic_keywords = {
        "Love and Romance": ["love", "marry", "marriage", "heart", "passion", "beloved", "affection"],
        "Social Class and Status": ["gentleman", "lady", "rank", "society", "station", "wealth", "poor"],
        "Family and Relationships": ["father", "mother", "sister", "brother", "family", "parent", "child"],
        "Morality and Ethics": ["moral", "virtue", "honor", "right", "wrong", "duty", "conscience"],
        "Wealth and Money": ["money", "fortune", "rich", "poor", "inheritance", "debt", "income"],
        "Justice and Law": ["justice", "law", "trial", "crime", "guilty", "innocent", "court"],
        "London and Urban Life": ["london", "city", "street", "house", "home", "village", "town"],
        "Secrets and Deception": ["secret", "hidden", "deceive", "truth", "lie", "reveal", "mystery"],
        "Innocence and Corruption": ["innocent", "pure", "corrupt", "ruin", "shame", "disgrace"],
        "Education and Knowledge": ["education", "learn", "study", "school", "knowledge", "wise"]
    }
    
    # Score topics based on keyword matches
    topic_scores = {}
    for theme, keywords in topic_keywords.items():
        score = sum(word_freq.get(kw, 0) for kw in keywords)
        if score > 0:
            topic_scores[theme] = score
    
    # Return top N topics
    top_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:num_topics]
    return [topic[0] for topic in top_topics]


def main():
    print("=" * 70)
    print("Multi-Author Text Preprocessing & Topic Extraction")
    print("=" * 70)
    
    # Define authors and their files
    authors_config = {
        "Dickens": {
            "files": ["data/raw/dickens_great_expectations.txt", "data/raw/dickens_a_tale_of_two_cities.txt"],
            "topics": []
        },
        "Austen": {
            "files": ["data/raw/austen_pride_and_prejudice.txt", "data/raw/austen_emma.txt"],
            "topics": []
        }
    }
    
    os.makedirs("data/clean", exist_ok=True)
    
    # Process each author
    for author_name, config in authors_config.items():
        print(f"\n{author_name}:")
        all_text = []
        
        for file_path in config["files"]:
            if not os.path.exists(file_path):
                print(f"  ✗ File not found: {file_path}")
                continue
            
            print(f"  Reading: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()
            
            cleaned = clean_gutenberg_text(raw)
            all_text.append(cleaned)
            print(f"    - Cleaned {len(cleaned)} characters")
        
        if not all_text:
            print(f"  ERROR: No files processed for {author_name}")
            continue
        
        # Combine all texts for this author
        combined_text = "\n\n".join(all_text)
        
        # Extract topics
        topics = extract_topics_from_text(combined_text, author_name, num_topics=7)
        config["topics"] = topics
        
        print(f"  Extracted Topics:")
        for i, topic in enumerate(topics, 1):
            print(f"    {i}. {topic}")
        
        # Save cleaned text
        output_file = f"data/clean/{author_name.lower()}_clean.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_text)
        print(f"  ✓ Saved cleaned text to: {output_file}")
    
    # Save topics configuration
    topics_config = {
        author: {
            "topics": config["topics"],
            "author_name": author
        }
        for author, config in authors_config.items()
    }
    
    with open("data/topics_config.json", "w") as f:
        json.dump(topics_config, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ Preprocessing complete!")
    print("✓ Topics configuration saved to: data/topics_config.json")
    print("\nNext step:")
    print("  python utils/build_class1_multi_author.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
