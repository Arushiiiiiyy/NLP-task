"""
Preprocessing script to:
1. Read Project Gutenberg .txt file (or epub converted to txt)
2. Clean the text (remove headers, footers, metadata)
3. Write clean text to data/clean/dickens_clean.txt
"""

import re
import os

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
    
    # Remove trailing footer
    text = re.sub(
        r"\*{3}.*?$",
        "",
        text,
        flags=re.DOTALL | re.MULTILINE
    )
    
    # Remove chapter headings
    text = re.sub(r"\n\s*(CHAPTER|Chapter|PART|Part)\s+[A-Z0-9IVX]+.*?\n", "\n", text)
    
   
    text = re.sub(r"\n\s*\n+", "\n\n", text)  
    text = re.sub(r"[ \t]+", " ", text)  
    
    
    text = re.sub(r"\[Page \d+\]", "", text)
    text = re.sub(r"\[Illustration:.*?\]", "", text)
    
    return text.strip()


def extract_paragraphs(text, min_words=100):
    """
    Split into paragraphs, keeping only those with sufficient content.
    """
    paragraphs = []
    for p in text.split("\n\n"):
        p = p.strip()
        if len(p.split()) >= min_words:
            paragraphs.append(p)
    return paragraphs


def main():
    """
    Main preprocessing pipeline.
    """
    
    raw_file = None
    for candidate in ["data/dickens_raw.txt", "data/dickens.txt"]:
        if os.path.exists(candidate):
            raw_file = candidate
            break
    
    if not raw_file:
        print("ERROR: Could not find raw Gutenberg text file.")
        print("Expected: data/dickens_raw.txt or data/dickens.txt")
        print("\nDownload from Project Gutenberg and save as: data/dickens_raw.txt")
        return
    
    
    print(f"Reading raw text from: {raw_file}")
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    print(f"Raw text length: {len(raw_text)} characters")
    
   
    print("Cleaning text...")
    cleaned_text = clean_gutenberg_text(raw_text)
    
    print(f"Cleaned text length: {len(cleaned_text)} characters")
    
    # Extract paragraphs
    print("Extracting paragraphs...")
    paragraphs = extract_paragraphs(cleaned_text, min_words=100)
    
    print(f"Extracted {len(paragraphs)} paragraphs")
    
    
    os.makedirs("data/clean", exist_ok=True)
    output_file = "data/clean/dickens_clean.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    
    print(f"\n✓ Cleaned text written to: {output_file}")
    print(f"✓ Ready for paragraph extraction with build_class1.py")
    
    
    if paragraphs:
        print(f"\nSample paragraph (first 100 words):")
        sample = " ".join(paragraphs[0].split()[:100])
        print(f"{sample}...\n")


if __name__ == "__main__":
    main()
