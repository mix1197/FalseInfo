from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load your trained model and vectorizer
model = joblib.load('models/model.pkl')  # Update with your model path
vectorizer = joblib.load('models/vectorizer.pkl')  # Update with your vectorizer path

@app.route('/false-information-checker')
def home():
    return render_template('index.html')

@app.route('/false-information-checker/predict', methods=['POST'])
def predict():
    try:
        # Get text from request
        text = request.json['text']
        
        # Transform text using vectorizer
        text_vector = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0].max()
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': bool(prediction),
            'prediction_label': 'Not Credible' if prediction else 'Credible',
            'confidence': float(probability)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)