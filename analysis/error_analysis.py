# analysis/error_analysis.py
misclassified = [(x,y,p) for x,y,p in zip(texts, labels, preds) if y != p]
