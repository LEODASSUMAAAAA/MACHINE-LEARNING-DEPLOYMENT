
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Define paths for model and scaler
MODEL_PATH = 'linear_regression_model_scaled.joblib'
SCALER_PATH = 'scaler.joblib'

# Load files dynamically
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler file not found on the server.'}), 500

    try:
        data = request.get_json(force=True)
        features = data.get('features')

        if not features or len(features) != 10:
            return jsonify({'error': 'Please provide exactly 10 feature values: [AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP, CO]'}), 400

        # Convert inputs to numpy array and shape for transformation
        features_array = np.array(features).reshape(1, -1)
        
        # Standardize the features using our saved scaler
        features_scaled = scaler.transform(features_array)
        
        # Make prediction with the scaled model
        prediction = model.predict(features_scaled)

        return jsonify({
            'prediction': float(prediction[0]),
            'target_variable': 'NOX'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'Linear Regression Model with Feature Scaling Flask API is running!',
        'required_features': ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'CO']
    })

if __name__ == '__main__':
    # Render binds to the PORT environment variable automatically
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
