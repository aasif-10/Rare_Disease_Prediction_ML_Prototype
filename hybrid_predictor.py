# hybrid_predictor.py
import os, joblib, pandas as pd, numpy as np
from models.disease_model import DiseasePredictor

# Load model + label encoder
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # project root
clf_path = os.path.join(ROOT_DIR, "models", "trained_disease_model.pkl")
le_path = os.path.join(ROOT_DIR, "models", "label_encoder.pkl")

clf = joblib.load(clf_path)
le = joblib.load(le_path)

# get trained feature names
feature_names = None
if hasattr(clf, "feature_names_in_") and clf.feature_names_in_ is not None:
    feature_names = list(clf.feature_names_in_)
else:
    feat_file = "models/feature_names.pkl"
    if os.path.exists(feat_file):
        feature_names = joblib.load(feat_file)

# fallback to rule-based (last resort)
rule_based = DiseasePredictor()
if feature_names is None:
    feature_names = rule_based.all_symptoms

print(f"[hybrid_predictor] using {len(feature_names)} features for ML input")

def hybrid_predict(text_symptoms=None, visual_symptoms=None):
    text_symptoms = text_symptoms or []
    visual_symptoms = visual_symptoms or []

    # Rule-based first
    rule_preds = rule_based.predict_diseases(text_symptoms, visual_symptoms)
    if rule_preds and rule_preds[0].get("probability", 0) > 0.6:
        return {"method": "rule-based", "predictions": rule_preds}

    # Build ML input matching exact training features
    row = {f: 0 for f in feature_names}
    for s in (text_symptoms + visual_symptoms):
        if s in row:
            row[s] = 1

    X_input = pd.DataFrame([row], columns=feature_names)

    try:
        proba = clf.predict_proba(X_input)[0]
        # predicted label index (encoded)
        pred_idx = clf.predict(X_input)[0]
    except Exception as e:
        # fallback to rule-based on error
        print("[hybrid_predictor] ML error:", e)
        return {"method": "rule-based", "predictions": rule_preds}

    ml_predictions = [{"disease": str(d), "probability": float(p)} for d, p in zip(le.classes_, proba)]
    ml_predictions.sort(key=lambda x: x["probability"], reverse=True)
    return {"method": "ml", "predictions": ml_predictions}
