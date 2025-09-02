from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import json
from werkzeug.utils import secure_filename
from PIL import Image

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.language_model import SimpleSymptomExtractor
from models.disease_model import DiseasePredictor
from models.vision_model import MedicalImageProcessor

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize AI models
print("🤖 Initializing AI Models...")
symptom_extractor = SimpleSymptomExtractor()
disease_predictor = DiseasePredictor()
image_processor = MedicalImageProcessor()
print("✅ All AI models loaded successfully!")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Rare Disease AI API is running",
        "models_loaded": True
    })

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """
    Analyze patient description text for symptoms
    
    Expected input:
    {
        "patient_description": "I bruise easily and am always tired"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'patient_description' not in data:
            return jsonify({
                "error": "Missing patient_description in request"
            }), 400
        
        patient_text = data['patient_description']
        
        # Extract symptoms using language model
        symptoms = symptom_extractor.extract_symptoms(patient_text)
        explanation = symptom_extractor.get_symptom_explanation(symptoms)
        
        return jsonify({
            "success": True,
            "patient_description": patient_text,
            "extracted_symptoms": symptoms,
            "explanation": explanation,
            "symptom_count": len(symptoms)
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Error analyzing text: {str(e)}"
        }), 500

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """
    Analyze medical image for visual symptoms
    """
    try:
        # Check if file is in request
        if 'image' not in request.files:
            return jsonify({
                "error": "No image file provided"
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                "error": "No file selected"
            }), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze image with mock AI
            visual_symptoms = image_processor.analyze_image_with_mock_ai(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                "success": True,
                "visual_symptoms": visual_symptoms,
                "message": f"Detected {len(visual_symptoms)} visual symptoms"
            })
        
        else:
            return jsonify({
                "error": "Invalid file type. Please upload PNG, JPG, JPEG, or GIF"
            }), 400
    
    except Exception as e:
        return jsonify({
            "error": f"Error analyzing image: {str(e)}"
        }), 500

@app.route('/api/predict-disease', methods=['POST'])
def predict_disease():
    """
    Predict diseases based on text and visual symptoms
    
    Expected input:
    {
        "text_symptoms": ["fatigue", "bruising"],
        "visual_symptoms": ["skin_hyperextensibility"]
    }
    """
    try:
        data = request.get_json()
        
        text_symptoms = data.get('text_symptoms', [])
        visual_symptoms = data.get('visual_symptoms', [])
        
        if not text_symptoms and not visual_symptoms:
            return jsonify({
                "error": "No symptoms provided"
            }), 400
        
        # Predict diseases
        predictions = disease_predictor.predict_diseases(
            text_symptoms=text_symptoms,
            visual_symptoms=visual_symptoms
        )
        
        # Get explanation for top prediction
        top_disease = predictions[0]['disease']
        explanation = disease_predictor.explain_prediction(
            top_disease, text_symptoms, visual_symptoms
        )
        
        return jsonify({
            "success": True,
            "text_symptoms": text_symptoms,
            "visual_symptoms": visual_symptoms,
            "predictions": predictions[:3],  # Top 3 predictions
            "top_prediction": predictions[0],
            "explanation": explanation
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Error predicting disease: {str(e)}"
        }), 500

@app.route('/api/complete-analysis', methods=['POST'])
def complete_analysis():
    """
    Complete analysis: text + image + disease prediction
    """
    try:
        # Get text data
        patient_description = request.form.get('patient_description', '')
        
        # Extract text symptoms
        text_symptoms = []
        if patient_description:
            text_symptoms = symptom_extractor.extract_symptoms(patient_description)
        
        # Analyze image if provided
        visual_symptoms = []
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                visual_symptoms = image_processor.analyze_image_with_mock_ai(filepath)
                os.remove(filepath)
        
        # Predict diseases
        predictions = disease_predictor.predict_diseases(
            text_symptoms=text_symptoms,
            visual_symptoms=visual_symptoms
        )
        
        # Get explanation
        if predictions:
            explanation = disease_predictor.explain_prediction(
                predictions[0]['disease'], text_symptoms, visual_symptoms
            )
        else:
            explanation = {"explanation": "No clear predictions available"}
        
        return jsonify({
            "success": True,
            "patient_description": patient_description,
            "text_symptoms": text_symptoms,
            "visual_symptoms": visual_symptoms,
            "predictions": predictions[:3],
            "explanation": explanation
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Error in complete analysis: {str(e)}"
        }), 500

@app.route('/api/available-symptoms', methods=['GET'])
def get_available_symptoms():
    """Get list of all symptoms the AI can detect"""
    return jsonify({
        "text_symptoms": list(symptom_extractor.symptom_patterns.keys()),
        "visual_symptoms": image_processor.visual_symptoms,
        "diseases": disease_predictor.diseases
    })

if __name__ == '__main__':
    print("🚀 Starting Rare Disease AI API...")
    print("Available endpoints:")
    print("  GET  /api/health")
    print("  POST /api/analyze-text")
    print("  POST /api/analyze-image") 
    print("  POST /api/predict-disease")
    print("  POST /api/complete-analysis")
    print("  GET  /api/available-symptoms")
    
    app.run(debug=True, port=5000)