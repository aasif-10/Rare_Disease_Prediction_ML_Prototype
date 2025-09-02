import sys
import os
import traceback
import sqlite3
import jwt
import datetime
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Add project root to path (parent of backend folder)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from functools import wraps

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
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'  # Change this in production

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize AI models
print("Initializing AI Models...")
symptom_extractor = SimpleSymptomExtractor()
image_processor = MedicalImageProcessor()
print("All AI models loaded successfully!")

# Database initialization
def init_db():
    conn = sqlite3.connect('diagnostic_portal.db')
    cursor = conn.cursor()
    
    # Create doctors table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL,
            text_symptoms TEXT,
            visual_symptoms TEXT,
            ai_predictions TEXT,
            ai_explanation TEXT,
            prescription TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert default doctor if not exists
    cursor.execute('SELECT * FROM doctors WHERE username = ?', ('doctor',))
    if not cursor.fetchone():
        cursor.execute('INSERT INTO doctors (username, password) VALUES (?, ?)', 
                      ('doctor', 'password123'))
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

# Initialize database on startup
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# JWT token decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            token = token.split(' ')[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_doctor = data['username']
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        
        return f(current_doctor, *args, **kwargs)
    return decorated

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "models_loaded": True,
        "message": "Rare Disease AI API is running with hybrid predictor"
    })

@app.route('/api/doctor/login', methods=['POST'])
def doctor_login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password required'}), 400
        
        # Check credentials in database
        conn = sqlite3.connect('diagnostic_portal.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM doctors WHERE username = ? AND password = ?', (username, password))
        doctor = cursor.fetchone()
        conn.close()
        
        if doctor:
            # Generate JWT token
            token = jwt.encode({
                'username': username,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }, app.config['SECRET_KEY'], algorithm="HS256")
            
            return jsonify({
                'success': True,
                'token': token,
                'message': 'Login successful'
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/doctor/patients', methods=['GET'])
@token_required
def get_patients(current_doctor):
    try:
        conn = sqlite3.connect('diagnostic_portal.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, patient_id, description, text_symptoms, visual_symptoms, 
                   ai_predictions, ai_explanation, prescription, created_at
            FROM patients 
            ORDER BY created_at DESC
        ''')
        patients = cursor.fetchall()
        conn.close()
        
        patient_list = []
        for patient in patients:
            patient_list.append({
                'id': patient[0],
                'patient_id': patient[1],
                'description': patient[2],
                'text_symptoms': patient[3].split(',') if patient[3] else [],
                'visual_symptoms': patient[4].split(',') if patient[4] else [],
                'ai_predictions': eval(patient[5]) if patient[5] else [],
                'ai_explanation': patient[6],
                'prescription': patient[7],
                'created_at': patient[8]
            })
        
        return jsonify({
            'success': True,
            'patients': patient_list
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/doctor/prescription/<int:patient_id>', methods=['POST'])
@token_required
def save_prescription(current_doctor, patient_id):
    try:
        data = request.get_json()
        prescription = data.get('prescription')
        
        if not prescription:
            return jsonify({'success': False, 'error': 'Prescription text required'}), 400
        
        conn = sqlite3.connect('diagnostic_portal.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE patients 
            SET prescription = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (prescription, patient_id))
        
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'success': False, 'error': 'Patient not found'}), 404
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Prescription saved successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patient/submit', methods=['POST'])
def patient_submit():
    try:
        print("Starting patient submission...")
        
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

        # Generate unique patient ID
        import uuid
        patient_id = str(uuid.uuid4())[:8].upper()
        
        # Save to database
        conn = sqlite3.connect('diagnostic_portal.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patients (patient_id, description, text_symptoms, visual_symptoms, 
                                ai_predictions, ai_explanation, prescription)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id,
            patient_description,
            ','.join(text_symptoms),
            ','.join(visual_symptoms),
            str(predictions),
            explanation['explanation'],
            ''  # Empty prescription initially
        ))
        conn.commit()
        conn.close()

        response_data = {
            "success": True,
            "patient_id": patient_id,
            "message": "Your case has been submitted successfully. A doctor will review it and provide a prescription."
        }

        print(f"Sending response: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        error_msg = f"Error in patient submission: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "success": False,
            "error": error_msg,
            "debug_traceback": traceback.format_exc()
        }), 500

@app.route('/api/patient/prescription/<patient_id>', methods=['GET'])
def get_patient_prescription(patient_id):
    try:
        conn = sqlite3.connect('diagnostic_portal.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT prescription, created_at, updated_at 
            FROM patients 
            WHERE patient_id = ?
        ''', (patient_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({'success': False, 'error': 'Patient not found'}), 404
        
        prescription, created_at, updated_at = result
        
        if not prescription:
            return jsonify({
                'success': True,
                'prescription': None,
                'message': 'No prescription available yet. Please check back later.',
                'created_at': created_at,
                'updated_at': updated_at
            })
        
        return jsonify({
            'success': True,
            'prescription': prescription,
            'created_at': created_at,
            'updated_at': updated_at
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    try:
        # Generate sample confusion matrix (replace with actual model evaluation)
        # In production, this should load from your trained model
        y_true = np.random.randint(0, 5, 100)  # Sample true labels
        y_pred = np.random.randint(0, 5, 100)  # Sample predicted labels
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create confusion matrix visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[f'Disease_{i}' for i in range(5)],
                    yticklabels=[f'Disease_{i}' for i in range(5)])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save to bytes buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Convert to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Extract metrics for each class
        metrics = {}
        for class_name, class_metrics in report.items():
            if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                metrics[class_name] = {
                    'precision': round(class_metrics['precision'], 3),
                    'recall': round(class_metrics['recall'], 3),
                    'f1_score': round(class_metrics['f1-score'], 3),
                    'support': int(class_metrics['support'])
                }
        
        return jsonify({
            'success': True,
            'image_base64': img_base64,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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
    print("  POST /api/patient/submit")
    print("  GET  /api/patient/prescription/<patient_id>")
    print("  POST /api/doctor/login")
    print("  GET  /api/doctor/patients")
    print("  POST /api/doctor/prescription/<patient_id>")
    print("  GET  /api/confusion-matrix")
    print("  GET  /api/test-hybrid")
    
    app.run(debug=True, port=5000)