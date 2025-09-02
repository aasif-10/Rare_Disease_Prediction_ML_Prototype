# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  # to save the trained model

# 1️⃣ Load dataset
data = pd.read_csv('data/rare_diseases_dataset.csv')

# 2️⃣ Split features and labels
X = data.drop('disease', axis=1)  # all symptom columns
y = data['disease']               # disease column

# 3️⃣ Encode disease labels to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4️⃣ Split into training and testing sets
# Removed 'stratify=y_encoded' because some classes have only 1 sample
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 5️⃣ Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6️⃣ Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, labels=range(len(le.classes_)), target_names=le.classes_
))

# 7️⃣ Save the trained model and label encoder
joblib.dump(clf, 'models/trained_disease_model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

print("\n✅ Model trained and saved successfully!")
