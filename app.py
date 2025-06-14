from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
from datetime import datetime
from model import CardioPredictor

app = Flask(__name__)
app.secret_key = 'cardio_predictor_secret_key_2024'

# Initialize the predictor
predictor = CardioPredictor()

# Load the trained model
if not predictor.load_model():
    print("No trained model found. Training new model...")
    from model import main
    main()
    predictor.load_model()

PREDICTIONS_FILE = 'predictions.json'

def load_predictions():
    """Load prediction history from JSON file"""
    if os.path.exists(PREDICTIONS_FILE):
        try:
            with open(PREDICTIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_prediction(prediction_data):
    """Save prediction to JSON file"""
    predictions = load_predictions()
    predictions.append(prediction_data)
    
    # Keep only last 100 predictions for performance
    if len(predictions) > 100:
        predictions = predictions[-100:]
    
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions, f, indent=2)

def validate_input(data):
    """Validate user input data"""
    errors = []
    
    try:
        age = float(data.get('age', 0))
        if age < 18 or age > 100:
            errors.append("Age must be between 18 and 100 years")
    except:
        errors.append("Invalid age format")
    
    try:
        height = float(data.get('height', 0))
        if height < 100 or height > 250:
            errors.append("Height must be between 100 and 250 cm")
    except:
        errors.append("Invalid height format")
    
    try:
        weight = float(data.get('weight', 0))
        if weight < 30 or weight > 200:
            errors.append("Weight must be between 30 and 200 kg")
    except:
        errors.append("Invalid weight format")
    
    try:
        ap_hi = int(data.get('ap_hi', 0))
        if ap_hi < 60 or ap_hi > 200:
            errors.append("Systolic blood pressure must be between 60 and 200 mmHg")
    except:
        errors.append("Invalid systolic blood pressure format")
    
    try:
        ap_lo = int(data.get('ap_lo', 0))
        if ap_lo < 40 or ap_lo > 120:
            errors.append("Diastolic blood pressure must be between 40 and 120 mmHg")
    except:
        errors.append("Invalid diastolic blood pressure format")
    
    try:
        if int(data.get('ap_hi', 0)) <= int(data.get('ap_lo', 0)):
            errors.append("Systolic pressure must be higher than diastolic pressure")
    except:
        pass
    
    # Validate categorical variables
    if data.get('gender') not in ['1', '2']:
        errors.append("Please select a valid gender")
    
    if data.get('cholesterol') not in ['1', '2', '3']:
        errors.append("Please select a valid cholesterol level")
    
    if data.get('gluc') not in ['1', '2', '3']:
        errors.append("Please select a valid glucose level")
    
    return errors

def get_risk_factors(user_input, dataset_stats):
    """Identify risk factors based on user input"""
    risk_factors = []
    
    if user_input['ap_hi'] > 140:
        risk_factors.append("High systolic blood pressure")
    
    if user_input['ap_lo'] > 90:
        risk_factors.append("High diastolic blood pressure")
    
    if user_input['cholesterol'] >= 2:
        risk_factors.append("Elevated cholesterol levels")
    
    if user_input['gluc'] >= 2:
        risk_factors.append("Elevated glucose levels")
    
    if user_input['smoke'] == 1:
        risk_factors.append("Smoking")
    
    if user_input['alco'] == 1:
        risk_factors.append("Alcohol consumption")
    
    if user_input['active'] == 0:
        risk_factors.append("Lack of physical activity")
    
    # BMI calculation
    height_m = user_input['height'] / 100
    bmi = user_input['weight'] / (height_m ** 2)
    if bmi > 30:
        risk_factors.append("Obesity (BMI > 30)")
    elif bmi > 25:
        risk_factors.append("Overweight (BMI > 25)")
    
    if user_input['age'] > 55:
        risk_factors.append("Advanced age")
    
    return risk_factors

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Validate input
        errors = validate_input(form_data)
        if errors:
            return render_template('index.html', errors=errors, form_data=form_data)
        
        # Prepare input for model
        user_input = {
            'age': float(form_data['age']),
            'gender': int(form_data['gender']),
            'height': float(form_data['height']),
            'weight': float(form_data['weight']),
            'ap_hi': int(form_data['ap_hi']),
            'ap_lo': int(form_data['ap_lo']),
            'cholesterol': int(form_data['cholesterol']),
            'gluc': int(form_data['gluc']),
            'smoke': int(form_data['smoke']),
            'alco': int(form_data['alco']),
            'active': int(form_data['active'])
        }
        
        # Make prediction
        result = predictor.predict(user_input)
        
        if result is None:
            return render_template('index.html', 
                                 error="Model prediction failed. Please try again.",
                                 form_data=form_data)
        
        # Get risk factors
        risk_factors = get_risk_factors(user_input, predictor.dataset_stats)
        
        # Calculate BMI for display
        height_m = user_input['height'] / 100
        bmi = round(user_input['weight'] / (height_m ** 2), 1)
        
        # Prepare result data
        result_data = {
            'prediction': result['prediction'],
            'risk_level': result['risk_level'],
            'probability': round(result['probability'] * 100, 1),
            'risk_factors': risk_factors,
            'bmi': bmi,
            'user_input': user_input
        }
        
        # Save prediction to history
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'input': user_input,
            'result': result,
            'risk_factors': risk_factors,
            'bmi': bmi
        }
        save_prediction(prediction_record)
        
        return render_template('index.html', result=result_data, form_data=form_data)
        
    except Exception as e:
        return render_template('index.html', 
                             error=f"An error occurred: {str(e)}", 
                             form_data=request.form.to_dict())

@app.route('/history')
def history():
    predictions = load_predictions()
    
    # Prepare data for charts
    chart_data = []
    for pred in predictions:
        chart_data.append({
            'timestamp': pred['timestamp'],
            'probability': pred['result']['probability'],
            'weight': pred['input']['weight'],
            'ap_hi': pred['input']['ap_hi'],
            'ap_lo': pred['input']['ap_lo'],
            'cholesterol': pred['input']['cholesterol'],
            'gluc': pred['input']['gluc'],
            'bmi': pred['bmi'],
            'risk_level': pred['result']['risk_level']
        })
    
    # Dataset averages for comparison
    dataset_stats = predictor.dataset_stats if hasattr(predictor, 'dataset_stats') else {}
    
    return render_template('history.html', 
                         predictions=predictions,
                         chart_data=chart_data,
                         dataset_stats=dataset_stats)

@app.route('/api/predictions')
def api_predictions():
    """API endpoint for prediction data"""
    predictions = load_predictions()
    return jsonify(predictions)

@app.route('/clear_history')
def clear_history():
    """Clear prediction history"""
    if os.path.exists(PREDICTIONS_FILE):
        os.remove(PREDICTIONS_FILE)
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)