# PRECOG: The Ghost in the Machine - Complete Project Guide

## Project Overview

This is a multi-task research project on **authorship attribution** and **AI text detection**. The project has 4 major tasks spanning dataset creation, feature analysis, ML modeling, and interpretability.

---

## Task 0: The Library of Babel
### Building the Dataset (Two Authors)

#### Goal
Create a dataset where the **primary variable is authorship, not topic**. 
- Class 1: Human-written text (Dickens + Austen novels from Project Gutenberg)
- Class 2: AI-generated paragraphs (Gemini API, generic style)
- Class 3: AI-generated paragraphs (Gemini API, author-specific style)

#### Setup Steps

**Step 1: Download texts from Project Gutenberg**
```bash
python download_texts.py
```
Downloads 4 novels (2 per author):
- Charles Dickens: *Great Expectations*, *A Tale of Two Cities*
- Jane Austen: *Pride and Prejudice*, *Emma*

**Step 2: Preprocess & Extract Topics**
```bash
python preprocess_multi_author.py
```
- Cleans raw Gutenberg text (removes headers, footers, formatting)
- Automatically extracts 5-10 core topics per author using keyword analysis
- Topics include: Love and Romance, Social Class, Family, Morality, Justice, Secrets, etc.
- Outputs: `data/clean/dickens_clean.txt`, `data/clean/austen_clean.txt`, `data/topics_config.json`

**Step 3: Extract Paragraphs**
```bash
python utils/build_class1_multi_author.py
```
- Extracts paragraphs (60-180 words) from cleaned texts
- Assigns topics deterministically
- Outputs: `data/class1.jsonl` (human paragraphs)

**Step 4: Generate AI Paragraphs (Class 2 & 3)**
```bash
# Set GEMINI_API_KEY environment variable first
export GEMINI_API_KEY="your-key-here"

python generate_gemini.py
```
- **Class 2**: Generic paragraphs on extracted topics (500 paragraphs)
- **Class 3**: Author-mimicking paragraphs (500 paragraphs)
- Both saved to `data/class2.jsonl` and `data/class3.jsonl`

#### Expected Output
```json
{
  "text": "In the heart of London's bustling streets...",
  "author": "Charles Dickens",
  "topic": "Social class and inequality",
  "class": 1,
  "source": "human"
}
```

---

## Task 1: The Fingerprint
### Proving Mathematical Distinctness

#### Analyses to Perform

**1. Lexical Richness**
- **Type-Token Ratio (TTR)**: `unique_words / total_words` (humans typically > AI)
- **Hapax Legomena**: Words appearing only once in 5,000-word sample

**2. Syntactic Complexity**
- **POS Distribution**: Ratio of Adjectives to Nouns (AI over-describes)
- **Dependency Tree Depth**: Average parse tree depth using SpaCy (longer = more complex)
- **Punctuation Density**: Heatmap of semicolons, em-dashes, exclamation marks

**3. Readability Indices**
- **Flesch-Kincaid Grade Level**: Sentence complexity metric

#### Files
- [task1/lexical.py](task1/lexical.py) - TTR, hapax legomena
- [task1/syntactic.py](task1/syntactic.py) - POS distribution, dependency depth
- [task1/punctuation.py](task1/punctuation.py) - Punctuation analysis
- [task1/readability.py](task1/readability.py) - Flesch-Kincaid

#### Run All Analyses
```bash
python task1/run_task1.py
```

#### Output
CSV file with all features:
```
text_id,author,class,ttr,hapax,adj_noun_ratio,dep_depth,flesch_kincaid,...
```

---

## Task 2: The Multi-Tiered Detective
### Building Three Classifiers

#### Tier A: The Statistician
**Model**: XGBoost/Random Forest
**Features**: Only numerical features from Task 1
**File**: [models/tierA_rf.py](models/tierA_rf.py)
```bash
python models/tierA_rf.py
```
Trains on: TTR, hapax, POS ratio, dependency depth, readability, punctuation counts

#### Tier B: The Semanticist
**Model**: Feedforward Neural Network
**Features**: Averaged pre-trained embeddings (GloVe/FastText)
**File**: [models/tierB_ffnn.py](models/tierB_ffnn.py)
```bash
python models/tierB_ffnn.py
```

#### Tier C: The Transformer
**Model**: LoRA fine-tuned distilbert-base-uncased
**File**: [models/tier_3_transformer.py](models/tier_3_transformer.py)
```bash
python models/tier_3_transformer.py
```
- Fine-tunes with LoRA (Low-Rank Adaptation)
- Checkpoint saved to: `models/tier_c/`

#### Expected Outputs
- Tier A: Accuracy, feature importance plot
- Tier B: Training curves, ROC-AUC
- Tier C: F1 score, confusion matrix

#### Note
If models fail to reach high accuracy, this is a **valid research finding**. Document why (e.g., AI may be learning human-like patterns, or humans may be repetitive like AI).

---

## Task 3: The Smoking Gun
### Explaining Model Predictions

#### Saliency Mapping with SHAP
**File**: [task3/saliency_shap.py](task3/saliency_shap.py)
```bash
python task3/saliency_shap.py
```

What it does:
- Loads best model (Tier C)
- Uses SHAP's PartitionExplainer
- Highlights words that most strongly signal "AI" to the model
- Generates interactive HTML visualization

#### Error Analysis
**File**: [task3/error_analysis.py](task3/error_analysis.py)
```bash
python task3/error_analysis.py
```

Find 3 examples where:
- **Human text was labeled AI**: Was it repetitive? Using uncommon words?
- **AI text was labeled Human**: Was it particularly eloquent or coherent?

#### Key Questions
1. Does the model pick up on specific "AI-isms"? (e.g., "tapestry," "delve," "testament")
2. Or is it detecting **sentence rhythm** and **flow patterns**?
3. Can we identify signature patterns unique to each author?

---

## Task 4: The Turing Test
### Adversarial Evolution of AI Text

#### Genetic Algorithm Approach
**File**: [ga/evolve.py](ga/evolve.py)
```bash
python ga/evolve.py
```

#### Workflow

1. **Initial Population**: Generate 10 "imposter" paragraphs using Gemini

2. **Fitness Function**: "Human" probability score from Tier C model
   - Target: Maximize human-like score

3. **Selection**: Keep top 3 paragraphs each generation

4. **Mutation**: Use Gemini to "perturb" winners:
   ```
   "Rewrite to change rhythm while keeping vocabulary"
   "Introduce subtle grammatical inconsistency or archaic word"
   ```

5. **Iteration**: Run for 5-10 generations

6. **Goal**: Achieve >90% "Human" confidence score

#### Personal Test
Take a piece of your own writing (SOP, essay) and run it through the detector:
- If labeled AI: Manually humanize it
- If labeled Human: Try to rewrite as LLM (overly helpful, structured)

---

## Dataset Structure

```
data/
├── raw/                          # Raw downloads from Project Gutenberg
│   ├── dickens_great_expectations.txt
│   ├── dickens_a_tale_of_two_cities.txt
│   ├── austen_pride_and_prejudice.txt
│   └── austen_emma.txt
│
├── clean/                        # Preprocessed texts
│   ├── dickens_clean.txt
│   ├── austen_clean.txt
│   └── dickens_clean.txt
│
├── class1.jsonl                  # Human paragraphs (Class 1)
├── class2.jsonl                  # Generic AI paragraphs (Class 2)
├── class3.jsonl                  # Author-style AI paragraphs (Class 3)
│
├── topics_config.json            # Extracted topics per author
└── task1_features.csv            # Features from Task 1 analysis
```

---

## File Organization

```
babel_folder/
├── download_texts.py             # Download from Gutenberg
├── preprocess_multi_author.py    # Clean & extract topics
├── generate_gemini.py            # Generate AI paragraphs
│
├── utils/
│   ├── build_class1_multi_author.py   # Extract human paragraphs
│   ├── feature_extraction.py
│   └── embeddings.py
│
├── task1/                        # Feature engineering
│   ├── run_task1.py
│   ├── lexical.py
│   ├── syntactic.py
│   ├── punctuation.py
│   └── readability.py
│
├── models/                       # Classifiers
│   ├── tierA_rf.py               # Random Forest (Task 2A)
│   ├── tierB_ffnn.py             # Neural Network (Task 2B)
│   ├── tier_3_transformer.py     # LoRA Transformer (Task 2C)
│   └── tier_c/                   # Saved checkpoints
│
├── task3/                        # Interpretability
│   ├── saliency_shap.py          # SHAP analysis
│   └── error_analysis.py         # Error breakdown
│
└── ga/                           # Task 4
    └── evolve.py                 # Genetic algorithm
```

---

## Dependencies

```
torch
transformers
peft
scikit-learn
xgboost
spacy
nltk
shap
captum
numpy
pandas
json
```

Install with:
```bash
pip install -r dependencies.txt
```

---

## Troubleshooting

### SHAP LoRA Loading Error
**Error**: "MISSING: those params were newly initialized"

**Solution**: This is **EXPECTED**. The model is fine-tuned for sequence classification, so it has different layer dimensions than the base model. The warning can be safely ignored.

### No paragraphs extracted
Check word count settings in `utils/build_class1_multi_author.py`:
```python
MIN_WORDS = 60      # Too high? Lower it
MAX_WORDS = 180     # Too low? Increase it
```

### Gemini API errors
Ensure API key is set:
```bash
export GEMINI_API_KEY="your-key"
```

---

## Research Questions

1. **Can we distinguish authorship (not topic)?**
   - Is there a measurable stylistic fingerprint?

2. **Can AI mimic human style?**
   - How similar is author-specific AI text to the original author?

3. **What signals AI?**
   - Are there specific word patterns? Punctuation habits? Sentence rhythm?

4. **Can we fool the detector?**
   - Using genetic algorithms, can we evolve text that bypasses classification?

5. **Is human writing detectably different?**
   - Or is modern AI good enough that the distinction is disappearing?

---

## Expected Timeline

- **Task 0**: 1-2 hours (download + preprocess)
- **Task 1**: 2-3 hours (feature extraction)
- **Task 2**: 4-6 hours (model training)
- **Task 3**: 2-3 hours (SHAP analysis)
- **Task 4**: 3-5 hours (GA evolution)

**Total**: ~15-20 hours of research and analysis

---

## Citation

**Project Title**: The Ghost in the Machine
**Principle**: "Le style, c'est l'homme même" (The style is the man himself) — Georges-Louis Leclerc

---

## Notes

- **Negative results are valid**: If models can't distinguish classes, document why thoroughly
- **Reproducibility**: Use deterministic seeds for all ML experiments
- **Human evaluation**: Consider manually reviewing samples flagged as "confused" by the model
- **Limitations**: Dataset limited to 2 authors; generalization to other writers unknown
