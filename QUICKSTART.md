# PRECOG Project - Quick Reference

## ✓ Current Status
- **Task 0**: ✓ COMPLETE - Dataset built (1280 human + 1000 AI paragraphs)
- **Task 1**: Ready to run
- **Task 2**: Ready to run
- **Task 3**: ✓ WORKING - SHAP saliency analysis fixed
- **Task 4**: Ready to run

---

## Run Each Task

### Task 1: Feature Analysis (The Fingerprint)
```bash
python task1/run_task1.py
```
Outputs: `data/task1_features.csv` with lexical, syntactic, readability metrics

### Task 2: Train Classifiers (The Multi-Tiered Detective)

**Tier A: Random Forest (numerical features)**
```bash
python models/tierA_rf.py
```

**Tier B: Feedforward NN (embeddings)**
```bash
python models/tierB_ffnn.py
```

**Tier C: LoRA Transformer**
```bash
python models/tier_3_transformer.py
```

### Task 3: Explain Predictions (The Smoking Gun)

**SHAP Saliency Mapping**
```bash
python task3/saliency_shap.py
```

**Error Analysis**
```bash
python task3/error_analysis.py
```

### Task 4: Fool the Detector (The Turing Test)

**Genetic Algorithm Evolution**
```bash
python ga/evolve.py
```

---

## Dataset Stats

| Class | Source | Count | Description |
|-------|--------|-------|-------------|
| 1 | Human | 1,280 | Dickens + Austen novels |
| 2 | AI (Generic) | 500 | Gemini API, generic style |
| 3 | AI (Author-style) | 500 | Gemini API, author-mimicking |
| **TOTAL** | | **2,280** | |

---

## Key Files

| File | Purpose |
|------|---------|
| `data/class1.jsonl` | Human paragraphs with author + topic |
| `data/class2.jsonl` | AI paragraphs (generic) |
| `data/class3.jsonl` | AI paragraphs (author-style) |
| `data/topics_config.json` | Extracted topics for each author |
| `data/task1_features.csv` | Task 1 feature matrix |
| `models/tier_c/` | Checkpoint for LoRA Transformer |

---

## Common Issues & Fixes

### Issue: "MISSING: those params were newly initialized"
**Status**: ✓ EXPECTED & FIXED
The SHAP script now properly handles LoRA adapter loading. This warning is harmless.

### Issue: No paragraphs extracted from clean text
**Fix**: 
```bash
# Check word count settings
grep -A 2 "MIN_WORDS\|MAX_WORDS" utils/build_class1_multi_author.py
```

### Issue: Gemini API errors
```bash
# Set your API key
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```


---

## Next Steps

1. Run **Task 1** to extract features: `python task1/run_task1.py`
2. Train **Task 2** classifiers: Start with Tier A (fastest)
3. Run **Task 3** SHAP analysis to understand predictions
4. Try **Task 4** genetic algorithm to fool the model

---

## Full Documentation

See `PROJECT_GUIDE.md` for detailed documentation of all 4 tasks.

---

**Status**: Ready for Tasks 1-4 execution ✓
