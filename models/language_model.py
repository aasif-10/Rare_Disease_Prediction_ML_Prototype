import torch
import torch.nn as nn
import json
import re

class MedicalLanguageModel(nn.Module):
    """
    This AI reads patient descriptions and extracts medical symptoms
    Like a smart medical translator
    """
    
    def __init__(self, vocab_size=1000, hidden_size=128):
        super(MedicalLanguageModel, self).__init__()
        
        # Word embedding: converts words to numbers AI can understand
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # LSTM: remembers context (like understanding "tired" after "always")
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Output layer: decides which medical symptoms are present
        self.output = nn.Linear(hidden_size, 20)  # 20 possible symptoms
        self.sigmoid = nn.Sigmoid()  # Converts to probabilities
    
    def forward(self, x):
        # Convert words to numbers
        embedded = self.embedding(x)
        
        # Process with memory (understand context)
        lstm_out, _ = self.lstm(embedded)
        
        # Take the last output (final understanding)
        final_output = lstm_out[:, -1, :]
        
        # Predict symptoms
        symptoms = self.sigmoid(self.output(final_output))
        
        return symptoms

class SimpleSymptomExtractor:
    """
    A simple rule-based approach that works well for hackathons
    Instead of complex AI training, uses medical keyword matching
    """
    
    def __init__(self):
        # Medical symptom keywords (what to look for in patient text)
        self.symptom_patterns = {
            "bruising": ["bruise", "bruising", "bruises", "easy bruising", "purple marks", "black and blue"],
            "fatigue": ["tired", "fatigue", "exhausted", "weak", "no energy", "always tired", "weakness"],
            "pallor": ["pale", "pallor", "white skin", "colorless", "washed out", "pale skin"],
            "shortness_of_breath": ["short of breath", "can't breathe", "breathing hard", "breathless", "out of breath"],
            "joint_pain": ["joint pain", "joints hurt", "arthritis", "stiff joints", "aching joints"],
            "skin_hyperextensibility": ["stretchy skin", "elastic skin", "skin stretches", "rubber skin"],
            "tachycardia": ["fast heartbeat", "racing heart", "heart pounds", "palpitations", "rapid heartbeat"],
            "orthostatic_intolerance": ["dizzy standing", "faint standing up", "dizzy when stand", "lightheaded standing"],
            "dizziness": ["dizzy", "lightheaded", "faint", "spinning", "vertigo"],
            "chest_pain": ["chest pain", "heart pain", "chest hurts", "chest ache"],
            "vision_problems": ["blurry vision", "can't see", "eye problems", "vision blurry", "poor vision"],
            "hearing_loss": ["can't hear", "hearing problems", "deaf", "hard of hearing", "hearing loss"],
            "bone_pain": ["bone pain", "bones hurt", "deep pain", "aching bones"],
            "easy_bleeding": ["bleed easily", "bleeding", "won't stop bleeding", "excessive bleeding"],
            "burning_pain": ["burning pain", "feels like fire", "burning sensation", "burning feeling"]
        }
    
    def extract_symptoms(self, patient_text):
        """
        Reads patient description and finds medical symptoms
        Like a smart medical dictionary lookup
        """
        
        # Convert to lowercase for easier matching
        text_lower = patient_text.lower()
        
        # Find matching symptoms
        found_symptoms = []
        
        for symptom, keywords in self.symptom_patterns.items():
            # Check if any keyword appears in patient text
            for keyword in keywords:
                if keyword in text_lower:
                    found_symptoms.append(symptom)
                    break  # Found this symptom, no need to check other keywords
        
        return found_symptoms
    
    def get_symptom_explanation(self, symptoms):
        """
        Creates human-readable explanation of found symptoms
        """
        if not symptoms:
            return "No specific symptoms detected in description."
        
        symptom_names = {
            "bruising": "Easy bruising",
            "fatigue": "Chronic fatigue", 
            "pallor": "Pale appearance",
            "shortness_of_breath": "Breathing difficulties",
            "joint_pain": "Joint pain",
            "skin_hyperextensibility": "Stretchy skin",
            "tachycardia": "Rapid heartbeat",
            "orthostatic_intolerance": "Dizziness when standing",
            "dizziness": "Dizziness",
            "chest_pain": "Chest pain",
            "vision_problems": "Vision problems",
            "hearing_loss": "Hearing loss",
            "bone_pain": "Bone pain",
            "easy_bleeding": "Easy bleeding",
            "burning_pain": "Burning pain"
        }
        
        readable_symptoms = [symptom_names.get(s, s.replace('_', ' ').title()) for s in symptoms]
        
        if len(readable_symptoms) == 1:
            return f"Detected symptom: {readable_symptoms[0]}"
        elif len(readable_symptoms) == 2:
            return f"Detected symptoms: {readable_symptoms[0]} and {readable_symptoms[1]}"
        else:
            return f"Detected symptoms: {', '.join(readable_symptoms[:-1])}, and {readable_symptoms[-1]}"

def test_language_model():
    """
    Test our symptom extraction on sample patient descriptions
    """
    
    print("Testing Language Model (Symptom Extraction)")
    print("=" * 50)
    
    extractor = SimpleSymptomExtractor()
    
    # Test cases
    test_cases = [
        "I bruise very easily and I'm always tired",
        "My skin looks really pale and I get short of breath when walking",  
        "I have terrible joint pain and my skin stretches like rubber",
        "My heart races when I stand up and I feel dizzy",
        "I have chest pain and blurry vision problems"
    ]
    
    for i, patient_text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Patient says: '{patient_text}'")
        
        # Extract symptoms
        symptoms = extractor.extract_symptoms(patient_text)
        explanation = extractor.get_symptom_explanation(symptoms)
        
        print(f"Detected symptoms: {symptoms}")
        print(f"Explanation: {explanation}")

if __name__ == "__main__":
    test_language_model()