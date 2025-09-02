import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Ensure root is in path

from hybrid_predictor import hybrid_predict

# Test symptoms
test_symptoms = ["fatigue", "joint_pain", "bruising"]

result = hybrid_predict(text_symptoms=test_symptoms)

print("Prediction method used:", result["method"])
print("Top 5 predictions:")
for pred in result["predictions"][:5]:
    print(pred)
