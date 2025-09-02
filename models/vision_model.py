import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import os

class MedicalVisionCNN(nn.Module):
    """
    AI that analyzes medical images for visual symptoms of rare diseases
    Detects: skin_hyperextensibility, pallor, joint_hypermobility, skin_spots, normal
    """
    
    def __init__(self):
        super(MedicalVisionCNN, self).__init__()
        
        # Layer 1: Basic pattern recognition
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Layer 2: Complex pattern recognition
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Layer 3: Advanced feature extraction
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Decision making layers
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 5)  # 5 visual symptoms
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Feature extraction
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten for classification
        x = x.view(-1, 64 * 28 * 28)
        
        # Classification layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class MedicalImageProcessor:
    """
    Processes images for rare disease symptom detection
    """
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Visual symptoms our AI can detect
        self.visual_symptoms = [
            "normal",
            "skin_hyperextensibility",  # EDS: stretchy skin
            "pallor",                   # General: pale appearance
            "joint_hypermobility",      # EDS/Marfan: flexible joints
            "skin_spots"                # Fabry: red/dark spots
        ]
    
    def process_image(self, image_path):
        """Process image for AI analysis"""
        try:
            image = Image.open(image_path).convert('RGB')
            processed_image = self.transform(image)
            return processed_image.unsqueeze(0)
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def analyze_image_with_mock_ai(self, image_path):
        """
        Mock AI analysis for demonstration
        In real implementation, this would use the trained model
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Mock analysis based on image properties
            # This simulates what a real AI would do
            avg_brightness = np.mean(image_array)
            color_variance = np.var(image_array)
            
            detected_symptoms = []
            
            # Simple heuristics for demo (real AI would be much more sophisticated)
            if avg_brightness < 100:
                detected_symptoms.append("pallor")
            
            if color_variance > 2000:
                detected_symptoms.append("skin_spots")
            
            # Random chance for other symptoms (for demo)
            if np.random.random() > 0.7:
                detected_symptoms.append("skin_hyperextensibility")
            
            if not detected_symptoms:
                detected_symptoms.append("normal")
            
            return detected_symptoms
        
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return ["normal"]

def test_vision_model():
    """Test the vision model"""
    print("Testing Vision Model for Rare Disease Detection...")
    
    model = MedicalVisionCNN()
    processor = MedicalImageProcessor()
    
    print(f"✅ Vision model created successfully!")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Can detect visual symptoms: {processor.visual_symptoms}")
    
    # Test with dummy data
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    probabilities = torch.softmax(output, dim=1)
    
    print("\n--- Sample Analysis (Dummy Data) ---")
    for i, symptom in enumerate(processor.visual_symptoms):
        prob = probabilities[0][i].item()
        print(f"{symptom}: {prob*100:.1f}%")
    
    return model, processor

if __name__ == "__main__":
    test_vision_model()