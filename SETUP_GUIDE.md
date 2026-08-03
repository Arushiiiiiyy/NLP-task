# PRECOG Task Setup Guide

## Issue: Human Paragraphs Not Being Written

**Root Cause**: Missing preprocessing step. The pipeline expected `data/clean/dickens_clean.txt` to exist, but no code was writing cleaned text to that file.

## Complete Workflow

### Step 1: Download Raw Text from Project Gutenberg
1. Visit https://www.gutenberg.org/
2. Search for "Charles Dickens" (or your chosen author)
3. Download the `.txt` file (UTF-8 encoding)
4. Save as `data/dickens_raw.txt`

Example:
```bash
curl -o data/dickens_raw.txt "https://www.gutenberg.org/cache/epub/730/pg730.txt"
```

### Step 2: Preprocess & Clean Text
Run the preprocessing script to clean and extract the text:
```bash
python preprocess_gutenberg.py
```

This will:
- ✓ Remove Project Gutenberg headers/footers
- ✓ Remove chapter headings
- ✓ Clean up formatting
- ✓ Write to `data/clean/dickens_clean.txt`

### Step 3: Extract Human Paragraphs
```bash
python utils/build_class1.py
```

This will:
- ✓ Read cleaned text from `data/clean/dickens_clean.txt`
- ✓ Extract paragraphs (60-180 words each)
- ✓ Assign topics deterministically
- ✓ Write to `data/class1.jsonl`

### Step 4: Verify Output
```bash
# Check how many paragraphs were extracted
wc -l data/class1.jsonl

# Sample a human paragraph
head -1 data/class1.jsonl | python -m json.tool
```

## Files Modified/Created

| File | Purpose |
|------|---------|
| `preprocess_gutenberg.py` | **NEW** - Cleans raw Gutenberg text |
| `utils/build_class1.py` | **UPDATED** - Better error handling |

## Expected Output Structure

```json
{
  "text": "Once upon a time in a small village...",
  "author": "Dickens",
  "topic": "Social class and inequality",
  "class": 1
}
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ERROR: Input file not found: data/dickens_raw.txt` | Download from Project Gutenberg and save to that path |
| `ERROR: Input file is empty` | Run `preprocess_gutenberg.py` first |
| No paragraphs extracted | Adjust `MIN_WORDS` and `MAX_WORDS` in `utils/build_class1.py` |

---

**Next Steps**: Once `class1.jsonl` is populated with human paragraphs, proceed with:
- `generate_gemini.py` to create AI paragraphs (Class 2 & 3)
- Task 1 feature extraction
- Task 2 model training


NOTE ------
export GEMINI_API_KEY='YOUR_GEMINI_API_KEY'

