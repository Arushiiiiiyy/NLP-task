"""
Download raw texts from Project Gutenberg for both authors.
This script downloads novels and saves them for preprocessing.
"""

import os
import urllib.request
import urllib.error


AUTHORS = {
    "dickens": {
        "name": "Charles Dickens",
        "novels": [
            {"id": 730, "title": "Great Expectations"},
            {"id": 1661, "title": "A Tale of Two Cities"},
        ]
    },
    "austen": {
        "name": "Jane Austen",
        "novels": [
            {"id": 1342, "title": "Pride and Prejudice"},
            {"id": 158, "title": "Emma"},
        ]
    }
}

def download_novel(gutenberg_id, author, title):
    """Download a single novel from Project Gutenberg."""
    url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
    
    os.makedirs("data/raw", exist_ok=True)
    filename = f"data/raw/{author}_{title.replace(' ', '_').lower()}.txt"
    
    if os.path.exists(filename):
        print(f"  ✓ Already exists: {filename}")
        return filename
    
    print(f"  Downloading: {title}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"  ✓ Saved to: {filename}")
        return filename
    except urllib.error.URLError as e:
        print(f"  ✗ Failed to download {title}: {e}")
        return None

def main():
    print("=" * 60)
    print("Downloading texts from Project Gutenberg")
    print("=" * 60)
    
    for author_key, author_data in AUTHORS.items():
        print(f"\n{author_data['name']}:")
        for novel in author_data['novels']:
            download_novel(novel['id'], author_key, novel['title'])
    
    print("\n" + "=" * 60)
    print("✓ Download complete! Next step:")
    print("  python preprocess_multi_author.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
