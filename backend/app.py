import sys
import os
import traceback

# Add project root to path (parent of backend folder)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import your hybrid predictor
from hybrid_predictor import hybrid_predict
from models.language_model import SimpleSymptomExtractor
from models.vision_model import MedicalImageProcessor

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize AI models
print("Initializing AI Models...")
symptom_extractor = SimpleSymptomExtractor()
image_processor = MedicalImageProcessor()
print("All AI models loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "models_loaded": True,
        "message": "Rare Disease AI API is running with hybrid predictor"
    })

@app.route('/api/complete-analysis', methods=['POST'])
def complete_analysis():
    try:
        print("Starting complete analysis...")
        
        # Extract text symptoms
        patient_description = request.form.get('patient_description', '')
        print(f"Patient description received: '{patient_description}'")
        
        text_symptoms = []
        if patient_description.strip():
            text_symptoms = symptom_extractor.extract_symptoms(patient_description)
            print(f"Text symptoms extracted: {text_symptoms}")

        # Extract visual symptoms
        visual_symptoms = []
        if 'image' in request.files:
            print("Processing uploaded image...")
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                try:
                    visual_symptoms = image_processor.analyze_image_with_mock_ai(filepath)
                    print(f"Visual symptoms extracted: {visual_symptoms}")
                except Exception as img_error:
                    print(f"Image processing error: {img_error}")
                    visual_symptoms = []
                finally:
                    # Clean up uploaded file
                    if os.path.exists(filepath):
                        os.remove(filepath)

        print(f"Total symptoms: Text={len(text_symptoms)}, Visual={len(visual_symptoms)}")

        # Use your hybrid predictor
        print("Calling hybrid predictor...")
        result = hybrid_predict(text_symptoms=text_symptoms, visual_symptoms=visual_symptoms)
        print(f"Hybrid predictor result: {result}")

        # Format predictions for frontend compatibility
        predictions = []
        for i, pred in enumerate(result["predictions"][:3]):  # Top 3
            # Ensure we have the required fields
            prediction = {
                "disease": str(pred.get("disease", "Unknown Disease")),
                "confidence": float(pred.get("probability", 0.0)),  # Keep as decimal for frontend conversion
                "matching_symptoms": len(text_symptoms + visual_symptoms),
                "total_disease_symptoms": 10  # Default value
            }
            predictions.append(prediction)
            
        print(f"Formatted predictions: {predictions}")

        explanation = {
            "explanation": f"Analysis completed using {result.get('method', 'hybrid')} method. "
                          f"Detected {len(text_symptoms)} text symptoms and {len(visual_symptoms)} visual symptoms."
        }

        response_data = {
            "success": True,
            "patient_description": patient_description,
            "text_symptoms": text_symptoms,
            "visual_symptoms": visual_symptoms,
            "predictions": predictions,
            "explanation": explanation,
            "method": result.get("method", "hybrid"),
            "debug_info": {
                "raw_predictions": result["predictions"][:3],
                "total_symptoms": len(text_symptoms + visual_symptoms)
            }
        }

        print(f"Sending response: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        error_msg = f"Error in analysis: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "success": False,
            "error": error_msg,
            "debug_traceback": traceback.format_exc()
        }), 500

@app.route('/api/test-hybrid', methods=['GET'])
def test_hybrid():
    """Test endpoint to verify hybrid predictor is working"""
    try:
        # Test with sample symptoms
        test_symptoms = ["fatigue", "joint_pain", "bruising"]
        result = hybrid_predict(text_symptoms=test_symptoms)
        
        return jsonify({
            "success": True,
            "test_symptoms": test_symptoms,
            "result": result,
            "message": "Hybrid predictor is working"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    print("Starting Rare Disease AI API on port 5000...")
    print("Using your trained hybrid predictor")
    print("Available endpoints:")
    print("  GET  /api/health")
    print("  POST /api/complete-analysis")
    print("  GET  /api/test-hybrid")
    
    app.run(debug=True, port=5000)