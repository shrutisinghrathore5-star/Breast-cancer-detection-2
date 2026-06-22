from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    features = np.array([[data[key] for key in sorted(data.keys())]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return jsonify({
        'diagnosis': 'benign' if prediction == 1 else 'malignant',
        'confidence': round(max(probability) * 100, 1)
    })

if __name__ == "__main__":
    app.run(debug=True)