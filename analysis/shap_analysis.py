# analysis/shap_analysis.py
import shap

explainer = shap.Explainer(model)
shap_values = explainer(texts)
shap.plots.text(shap_values[0])
