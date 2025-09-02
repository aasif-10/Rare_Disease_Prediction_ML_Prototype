import pandas as pd
import numpy as np

# Define diseases and their common symptoms
diseases = {
    "Ehlers-Danlos Syndrome": ["fatigue","joint_pain","bruising","skin_hyperextensibility","joint_hypermobility","easy_bleeding"],
    "POTS": ["tachycardia","orthostatic_intolerance","dizziness","fatigue","pallor"],
    "Marfan Syndrome": ["joint_pain","vision_problems","chest_pain","joint_hypermobility"],
    "Fabry Disease": ["burning_pain","hearing_loss","skin_spots"],
    "Gaucher Disease": ["fatigue","bone_pain","easy_bleeding","bruising"],
    "Flu": ["fatigue","chest_pain","nausea"],
    "Diabetes": ["fatigue","vision_problems","nausea"],
    "Common Cold": ["fatigue","dizziness","chest_pain"],
    "Hypertension": ["chest_pain","dizziness","tachycardia"],
    "Anemia": ["fatigue","pallor","dizziness"],
    "Huntington Disease": ["fatigue","bone_pain"],
    "Wilson Disease": ["fatigue","vision_problems","pallor"],
    "Tay-Sachs Disease": ["fatigue","hearing_loss","vision_problems"],
    "Pompe Disease": ["fatigue","joint_pain","bone_pain"],
    "Niemann-Pick Disease": ["fatigue","bruising","bone_pain"],
    "Alkaptonuria": ["fatigue","joint_pain","bone_pain"],
    "Prader-Willi Syndrome": ["fatigue","joint_pain"],
    "Angelman Syndrome": ["fatigue","dizziness"],
    "Rett Syndrome": ["fatigue","joint_pain"],
    "Cystic Fibrosis": ["fatigue","pallor","chest_pain"]
}

all_symptoms = ["fatigue","joint_pain","tachycardia","bruising","skin_hyperextensibility",
                "joint_hypermobility","orthostatic_intolerance","dizziness","vision_problems",
                "chest_pain","burning_pain","hearing_loss","bone_pain","easy_bleeding",
                "skin_spots","pallor","nausea"]

rows = []

samples_per_disease = 100

for disease, common_symptoms in diseases.items():
    for _ in range(samples_per_disease):
        sample = {}
        sample['disease'] = disease
        for symptom in all_symptoms:
            # 70% chance of assigning common symptom, 5% for others
            if symptom in common_symptoms:
                sample[symptom] = np.random.choice([1,0], p=[0.7,0.3])
            else:
                sample[symptom] = np.random.choice([1,0], p=[0.05,0.95])
        rows.append(sample)

# Create DataFrame
df = pd.DataFrame(rows)

# Save as CSV
df.to_csv("rare_diseases_dataset.csv", index=False)
print("✅ Dataset generated: rare_diseases_dataset.csv")
