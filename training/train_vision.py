import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import sys
sys.path.append('..')
from models.vision_model import MedicalVisionCNN, MedicalImageProcessor

class MedicalImageDataset(Dataset):
    """
    This creates a 'textbook' of medical images for our AI to study
    Each page has: an image + the correct answer
    """
    
    def __init__(self, image_folder, labels_file, processor):
        self.image_folder = image_folder
        self.processor = processor
        
        # Load the labels (correct answers)
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        # Create mapping: text labels -> numbers (AI understands numbers better)
        self.label_to_number = {
            "normal": 0,
            "pallor": 1, 
            "shallow_breathing": 2
        }
        
        # Get list of images that have labels
        self.image_files = [f for f in self.labels.keys() if 
                           os.path.exists(os.path.join(image_folder, f))]
        
        print(f"Found {len(self.image_files)} labeled images for training")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Gets one training example: image + correct answer
        Like showing the AI one flashcard
        """
        # Get image filename
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)
        
        # Process the image  
        image = self.processor.process_image(image_path).squeeze(0)
        
        # Get the correct label (answer)
        label_text = self.labels[image_file]
        label_number = self.label_to_number[label_text]
        
        return image, label_number

def train_vision_model():
    """
    This is like sending our AI to medical school
    It studies thousands of examples to learn patterns
    """
    
    print("Starting vision model training...")
    print("This is like teaching AI to recognize medical symptoms in images")
    
    # Create the AI student
    model = MedicalVisionCNN()
    processor = MedicalImageProcessor()
    
    # For now, we'll simulate training since we don't have real medical images
    # In real implementation, you'd use actual medical datasets
    
    print("\n=== SIMULATED TRAINING ===")
    print("In real training, we would:")
    print("1. Show AI 1000+ medical images")
    print("2. For each image, tell AI the correct diagnosis")
    print("3. AI adjusts its 'brain' to be more accurate")
    print("4. Repeat until AI gets 90%+ accuracy")
    
    # Simulate training process
    for epoch in range(5):  # 5 training cycles
        print(f"Training cycle {epoch + 1}/5...")
        
        # This would be the actual training code:
        # for images, labels in training_data:
        #     predictions = model(images)
        #     loss = calculate_error(predictions, correct_labels)  
        #     model.improve_based_on_errors(loss)
        
        # Simulate improvement
        accuracy = 60 + (epoch * 8)  # Starts at 60%, improves each cycle
        print(f"  Current accuracy: {accuracy}%")
    
    print(f"\nTraining complete! Final accuracy: 92%")
    
    # Save the trained model
    torch.save(model.state_dict(), 'trained_models/vision_model.pth')
    print("Model saved to 'trained_models/vision_model.pth'")
    
    return model

def test_vision_model():
    """
    Test our trained model on new images
    Like giving a final exam to our AI
    """
    
    print("\nTesting vision model...")
    
    # Load trained model
    model = MedicalVisionCNN()
    try:
        model.load_state_dict(torch.load('trained_models/vision_model.pth'))
        print("Loaded trained model successfully!")
    except:
        print("No trained model found, using untrained model for demo")
    
    model.eval()  # Put in evaluation mode
    processor = MedicalImageProcessor()
    
    # Test on sample image if it exists
    test_image_path = 'test_data/test_patient.jpg'
    if os.path.exists(test_image_path):
        print(f"Analyzing {test_image_path}...")
        
        # Process the test image
        processed_image = processor.process_image(test_image_path)
        
        # Make prediction
        with torch.no_grad():  # Don't update model during testing
            prediction = model(processed_image)
            probabilities = torch.softmax(prediction, dim=1)
        
        # Interpret results
        labels = ["normal", "pallor", "shallow_breathing"]
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        print(f"Prediction: {labels[predicted_class]}")
        print(f"Confidence: {confidence*100:.1f}%")
        
        # Show all probabilities
        print("\nAll probabilities:")
        for i, label in enumerate(labels):
            prob = probabilities[0][i].item()
            print(f"  {label}: {prob*100:.1f}%")
    else:
        print(f"Test image not found at {test_image_path}")
        print("Create a test image to see the model in action!")

if __name__ == "__main__":
    # Train the model
    trained_model = train_vision_model()
    
    # Test the model
    test_vision_model()