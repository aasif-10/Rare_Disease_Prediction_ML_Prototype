# train_model_enhanced.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1️⃣ Load the enhanced dataset
data = pd.read_csv('rare_diseases_dataset_enhanced.csv')

# 2️⃣ Split features and labels
X = data.drop('disease', axis=1)  # all symptom + category columns
y = data['disease']               # target column

# Save feature names for inference
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'models/feature_names_enhanced.pkl')
print("Saved feature names (count={}):".format(len(feature_names)))

# 3️⃣ Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4️⃣ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5️⃣ Train Random Forest Classifier with more estimators
clf = RandomForestClassifier(
    n_estimators=300,   # increase trees from 100 -> 300
    max_depth=None,     # let trees grow fully
    random_state=42,
    n_jobs=-1           # use all CPU cores
)
clf.fit(X_train, y_train)

# 6️⃣ Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, labels=range(len(le.classes_)), target_names=le.classes_
))

# 7️⃣ Save the trained model and label encoder
joblib.dump(clf, 'models/trained_disease_model_enhanced.pkl')
joblib.dump(le, 'models/label_encoder_enhanced.pkl')

print("\n✅ Enhanced model trained and saved successfully!")
