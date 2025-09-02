import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import json
import os

class MedicalVisionCNN(nn.Module):
    """
    This is our AI 'doctor' that looks at medical images
    Think of it as having layers of recognition:
    - Layer 1: Spots basic shapes and colors
    - Layer 2: Recognizes medical patterns  
    - Layer 3: Makes final diagnosis decision
    """
    
    def __init__(self):
        super(MedicalVisionCNN, self).__init__()
        
        # Layer 1: Basic pattern recognition (like recognizing edges)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Layer 2: Complex pattern recognition (like recognizing skin texture)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Layer 3: Decision making (combines all patterns to make diagnosis)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # 32 patterns, 56x56 size each
        self.fc2 = nn.Linear(128, 3)  # 3 possible outputs: pallor, normal, shallow_breathing
        
        self.dropout = nn.Dropout(0.5)  # Prevents overfitting (like not memorizing)
    
    def forward(self, x):
        """
        This is the 'thinking process' of our AI doctor
        x = the input image
        """
        # Step 1: Look for basic patterns
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Step 2: Look for complex patterns  
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Step 3: Flatten the image data for decision making
        x = x.view(-1, 32 * 56 * 56)
        
        # Step 4: Make intermediate decision
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Step 5: Final diagnosis decision
        x = self.fc2(x)
        
        return x

class MedicalImageProcessor:
    """
    This prepares images for our AI to analyze
    Like cleaning glasses before looking through them
    """
    
    def __init__(self):
        # These are the steps to prepare any image for our AI
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),          # Make all images same size
            transforms.ToTensor(),                   # Convert to numbers AI can understand
            transforms.Normalize(                    # Adjust colors to standard range
                mean=[0.485, 0.456, 0.406],        # Standard values for medical images
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def process_image(self, image_path):
        """
        Takes an image file and prepares it for AI analysis
        """
        # Load the image
        image = Image.open(image_path).convert('RGB')
        
        # Apply all the preparation steps
        processed_image = self.transform(image)
        
        # Add batch dimension (AI expects multiple images, even if we have 1)
        processed_image = processed_image.unsqueeze(0)
        
        return processed_image
def test_vision_model():
    print("Testing Vision Model...")
    print("Creating AI that can analyze medical images...")
    
    model = MedicalVisionCNN()
    processor = MedicalImageProcessor()
    
    print(f"✅ Vision model created successfully!")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print("This AI can detect: pallor, normal skin, shallow breathing")
    
    return model

if __name__ == "__main__":
    test_vision_model()


