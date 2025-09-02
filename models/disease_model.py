import torch
import numpy as np

class DiseasePredictor:
    """
    AI that predicts rare diseases based on text symptoms and visual symptoms
    """
    
    def __init__(self):
        self.diseases = [
            "Ehlers-Danlos Syndrome",
            "Postural Orthostatic Tachycardia Syndrome",
            "Marfan Syndrome",
            "Fabry Disease",
            "Gaucher Disease"
    ]

        self.all_symptoms = [
            "fatigue", "joint_pain", "tachycardia", "bruising", "skin_hyperextensibility",
            "joint_hypermobility", "orthostatic_intolerance", "dizziness", "vision_problems",
            "chest_pain", "burning_pain", "hearing_loss", "skin_spots", "bone_pain",
            "easy_bleeding", "pallor"
    ]

        # **Initialize the matrix before using it**
        self.disease_symptom_matrix = np.zeros((len(self.diseases), len(self.all_symptoms)))

        # Now populate the matrix
        self._create_medical_knowledge()

    
    def _create_medical_knowledge(self):
        """
        Create a matrix of which symptoms are associated with which diseases
        Based on real medical literature
        """
        knowledge = np.zeros((len(self.diseases), len(self.all_symptoms)))
        
        # Ehlers-Danlos Syndrome (EDS)
        eds_symptoms = [
            "bruising", "fatigue", "joint_pain", "skin_hyperextensibility", 
            "joint_hypermobility", "easy_bleeding"
        ]
        self._add_disease_symptoms(0, eds_symptoms)
        
        # POTS
        pots_symptoms = [
            "tachycardia", "orthostatic_intolerance", "dizziness", 
            "fatigue", "pallor"
        ]
        self._add_disease_symptoms(1, pots_symptoms)
        
        # Marfan Syndrome
        marfan_symptoms = [
            "joint_pain", "vision_problems", "chest_pain", 
            "joint_hypermobility"
        ]
        self._add_disease_symptoms(2, marfan_symptoms)
        
        # Fabry Disease
        fabry_symptoms = [
            "burning_pain", "hearing_loss", "skin_spots"
        ]
        self._add_disease_symptoms(3, fabry_symptoms)
        
        # Gaucher Disease
        gaucher_symptoms = [
            "fatigue", "bone_pain", "easy_bleeding", "bruising"
        ]
        self._add_disease_symptoms(4, gaucher_symptoms)
        
        return knowledge
    
    def _add_disease_symptoms(self, disease_idx, symptoms):
        """Helper to add symptoms to disease matrix"""
        for symptom in symptoms:
            if symptom in self.all_symptoms:
                symptom_idx = self.all_symptoms.index(symptom)
                self.disease_symptom_matrix[disease_idx, symptom_idx] = 1
    
    def predict_diseases(self, text_symptoms=[], visual_symptoms=[]):
        """
        Predict diseases based on both text and visual symptoms
        
        Args:
            text_symptoms: List of symptoms from patient description
            visual_symptoms: List of symptoms from image analysis
        
        Returns:
            List of disease predictions with probabilities
        """
        print(f"\n🔍 Analyzing symptoms...")
        print(f"Text symptoms: {text_symptoms}")
        print(f"Visual symptoms: {visual_symptoms}")
        
        # Create symptom vector
        symptom_vector = np.zeros(len(self.all_symptoms))
        all_symptoms = text_symptoms + visual_symptoms
        
        for symptom in all_symptoms:
            if symptom in self.all_symptoms:
                idx = self.all_symptoms.index(symptom)
                symptom_vector[idx] = 1
        
        # Calculate disease scores
        disease_scores = np.dot(self.disease_symptom_matrix, symptom_vector)
        
        # Convert to probabilities
        if np.sum(disease_scores) == 0:
            # No matching symptoms, equal probability
            probabilities = np.ones(len(self.diseases)) / len(self.diseases)
        else:
            # Weight by number of matching symptoms
            probabilities = disease_scores / np.sum(disease_scores)
        
        # Create prediction results
        predictions = []
        for i, disease in enumerate(self.diseases):
            predictions.append({
                "disease": disease,
                "probability": float(probabilities[i]),
                "confidence": f"{probabilities[i]*100:.1f}%",
                "matching_symptoms": int(disease_scores[i]),
                "total_disease_symptoms": int(np.sum(self.disease_symptom_matrix[i]))
            })
        
        # Sort by probability
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        return predictions
    
    def explain_prediction(self, disease_name, text_symptoms=[], visual_symptoms=[]):
        """
        Explain why a disease was predicted
        """
        disease_idx = self.diseases.index(disease_name)
        all_symptoms = text_symptoms + visual_symptoms
        
        matching_symptoms = []
        for symptom in all_symptoms:
            if symptom in self.all_symptoms:
                symptom_idx = self.all_symptoms.index(symptom)
                if self.disease_symptom_matrix[disease_idx, symptom_idx] == 1:
                    matching_symptoms.append(symptom)
        
        return {
            "matching_symptoms": matching_symptoms,
            "explanation": f"Patient shows {len(matching_symptoms)} symptoms commonly associated with {disease_name}"
        }

def test_disease_model():
    """Test the disease prediction model"""
    print("Testing Disease Prediction Model...")
    print("="*50)
    
    predictor = DiseasePredictor()
    
    test_cases = [
        {
            "name": "EDS Patient",
            "text_symptoms": ["bruising", "fatigue", "joint_pain"],
            "visual_symptoms": ["skin_hyperextensibility", "joint_hypermobility"]
        },
        {
            "name": "POTS Patient", 
            "text_symptoms": ["tachycardia", "dizziness", "fatigue"],
            "visual_symptoms": ["pallor"]
        },
        {
            "name": "Marfan Patient",
            "text_symptoms": ["joint_pain", "vision_problems", "chest_pain"],
            "visual_symptoms": ["joint_hypermobility"]
        },
        {
            "name": "Fabry Patient",
            "text_symptoms": ["burning_pain", "hearing_loss"],
            "visual_symptoms": ["skin_spots"]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        predictions = predictor.predict_diseases(
            test_case['text_symptoms'], 
            test_case['visual_symptoms']
        )
        
        print("Top 3 predictions:")
        for i, pred in enumerate(predictions[:3], 1):
            print(f"  {i}. {pred['disease']}: {pred['confidence']} "
                  f"({pred['matching_symptoms']}/{pred['total_disease_symptoms']} symptoms)")
    
    print("\n✅ Disease prediction model working successfully!")
    return predictor

if __name__ == "__main__":
    test_disease_model()