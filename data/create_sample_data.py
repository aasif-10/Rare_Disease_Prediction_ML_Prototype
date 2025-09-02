# This creates fake training data for testing
# In real life, you'd use actual medical datasets

import os
import json

def create_sample_image_labels():
    """Creates labels for images (what each image shows)"""
    
    # Simulating 100 medical images with labels
    image_labels = {
        "patient_001.jpg": "pallor",      # This image shows pale skin
        "patient_002.jpg": "normal",      # This image shows normal skin  
        "patient_003.jpg": "pallor",
        "patient_004.jpg": "shallow_breathing",
        "patient_005.jpg": "normal",
        # ... we'd have 100+ real entries
    }
    
    # Save to file
    with open('data/image_labels.json', 'w') as f:
        json.dump(image_labels, f)
    
    print("Created sample image labels!")

def create_sample_symptom_data():
    """Creates sample patient descriptions with medical terms"""
    
    # Patient descriptions paired with medical symptoms
    symptom_data = [
        {
            "patient_text": "I bruise very easily and feel tired all the time",
            "medical_symptoms": ["bruising", "fatigue"]
        },
        {
            "patient_text": "My skin looks very pale and I get short of breath",
            "medical_symptoms": ["pallor", "shortness_of_breath"]  
        },
        {
            "patient_text": "I have joint pain and my skin stretches a lot",
            "medical_symptoms": ["joint_pain", "skin_hyperextensibility"]
        },
        {
            "patient_text": "My heart races when I stand up and I feel dizzy",
            "medical_symptoms": ["tachycardia", "orthostatic_intolerance"]
        },
        # We'd have 1000+ real entries
    ]
    
    # Save to file
    with open('data/symptom_training.json', 'w') as f:
        json.dump(symptom_data, f)
        
    print("Created sample symptom data!")

def create_disease_symptom_connections():
    """Creates the knowledge of which symptoms connect to which diseases"""
    
    # This is medical knowledge: which diseases have which symptoms
    disease_connections = {
        "Ehlers-Danlos": {
            "symptoms": ["bruising", "fatigue", "joint_pain", "skin_hyperextensibility"],
            "visual_signs": ["pallor", "skin_transparency"]
        },
        "POTS": {
            "symptoms": ["tachycardia", "orthostatic_intolerance", "fatigue", "dizziness"],
            "visual_signs": ["pallor"]
        },
        "Marfan": {
            "symptoms": ["joint_pain", "vision_problems", "chest_pain"],
            "visual_signs": ["tall_stature", "long_fingers"]
        },
        "Fabry": {
            "symptoms": ["burning_pain", "fatigue", "hearing_loss"],
            "visual_signs": ["angiokeratoma", "corneal_deposits"]
        },
        "Gaucher": {
            "symptoms": ["fatigue", "bone_pain", "easy_bleeding"],
            "visual_signs": ["enlarged_spleen", "yellowing"]
        }
    }
    
    # Save to file  
    with open('data/disease_knowledge.json', 'w') as f:
        json.dump(disease_connections, f)
        
    print("Created disease knowledge base!")

if __name__ == "__main__":
    create_sample_image_labels()
    create_sample_symptom_data()  
    create_disease_symptom_connections()
    print("\nAll sample data created! Ready for training.")