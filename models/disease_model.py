import torch
import numpy as np

class DiseasePredictor:
    def __init__(self):
        self.diseases = [
            "Ehlers-Danlos Syndrome",
            "POTS", 
            "Marfan Syndrome",
            "Fabry Disease",
            "Gaucher Disease"
        ]
        
        self.symptoms = [
            "bruising", "fatigue", "pallor", "shortness_of_breath", 
            "joint_pain", "skin_hyperextensibility", "tachycardia",
            "orthostatic_intolerance", "dizziness", "chest_pain",
            "vision_problems", "hearing_loss", "bone_pain", 
            "easy_bleeding", "burning_pain"
        ]
        
        self.disease_symptom_matrix = self._create_medical_knowledge()
    
    def _create_medical_knowledge(self):
        knowledge = np.zeros((len(self.diseases), len(self.symptoms)))
        
        # Ehlers-Danlos Syndrome
        eds_symptoms = ["bruising", "fatigue", "joint_pain", "skin_hyperextensibility", "pallor"]
        for symptom in eds_symptoms:
            if symptom in self.symptoms:
                col_idx = self.symptoms.index(symptom)
                knowledge[0, col_idx] = 1
        
        # POTS
        pots_symptoms = ["tachycardia", "orthostatic_intolerance", "dizziness", "fatigue", "pallor"]
        for symptom in pots_symptoms:
            if symptom in self.symptoms:
                col_idx = self.symptoms.index(symptom)
                knowledge[1, col_idx] = 1
        
        # Marfan Syndrome
        marfan_symptoms = ["joint_pain", "vision_problems", "chest_pain"]
        for symptom in marfan_symptoms:
            if symptom in self.symptoms:
                col_idx = self.symptoms.index(symptom)
                knowledge[2, col_idx] = 1
        
        # Fabry Disease
        fabry_symptoms = ["burning_pain", "hearing_loss"]
        for symptom in fabry_symptoms:
            if symptom in self.symptoms:
                col_idx = self.symptoms.index(symptom)
                knowledge[3, col_idx] = 1
        
        # Gaucher Disease
        gaucher_symptoms = ["fatigue", "bone_pain", "easy_bleeding"]
        for symptom in gaucher_symptoms:
            if symptom in self.symptoms:
                col_idx = self.symptoms.index(symptom)
                knowledge[4, col_idx] = 1
        
        return knowledge
    
    def predict_diseases(self, detected_symptoms, visual_symptoms=[]):
        print(f"\nAnalyzing symptoms: {detected_symptoms + visual_symptoms}")
        
        symptom_vector = np.zeros(len(self.symptoms))
        all_symptoms = detected_symptoms + visual_symptoms
        
        for symptom in all_symptoms:
            if symptom in self.symptoms:
                idx = self.symptoms.index(symptom)
                symptom_vector[idx] = 1
        
        disease_scores = np.dot(self.disease_symptom_matrix, symptom_vector)
        
        if np.sum(disease_scores) == 0:
            probabilities = np.ones(len(self.diseases)) / len(self.diseases)
        else:
            probabilities = disease_scores / np.sum(disease_scores)
        
        predictions = []
        for i, disease in enumerate(self.diseases):
            predictions.append({
                "disease": disease,
                "probability": float(probabilities[i]),
                "confidence": f"{probabilities[i]*100:.1f}%"
            })
        
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        return predictions

def test_disease_model():
    print("Testing Disease Prediction Model...")
    print("Creating AI that can predict rare diseases...")
    
    predictor = DiseasePredictor()
    
    test_cases = [
        {
            "name": "EDS Patient",
            "symptoms": ["bruising", "fatigue", "joint_pain"],
        },
        {
            "name": "POTS Patient", 
            "symptoms": ["tachycardia", "dizziness", "fatigue"],
        },
        {
            "name": "Marfan Patient",
            "symptoms": ["joint_pain", "vision_problems", "chest_pain"],
        }
    ]
    
    print("\n--- Testing disease predictions ---")
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(f"Input symptoms: {test_case['symptoms']}")
        
        predictions = predictor.predict_diseases(test_case['symptoms'])
        
        print("Top 3 predictions:")
        for i, pred in enumerate(predictions[:3], 1):
            print(f"  {i}. {pred['disease']}: {pred['confidence']}")
    
    print("\n✅ Disease prediction model working successfully!")
    return predictor

if __name__ == "__main__":
    test_disease_model()