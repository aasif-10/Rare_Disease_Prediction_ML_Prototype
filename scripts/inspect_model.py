# scripts/inspect_model.py
import joblib, os
p = "models/trained_disease_model.pkl"
if not os.path.exists(p):
    print("ERROR: models/trained_disease_model.pkl not found")
else:
    clf = joblib.load(p)
    print("n_features_in_:", getattr(clf, "n_features_in_", None))
    print("feature_names_in_:", getattr(clf, "feature_names_in_", None))
