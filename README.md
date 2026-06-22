# Breast Cancer Detection Web Application

## Overview

This project is a machine learning-powered web application that predicts whether a breast tumor is **Benign** or **Malignant** based on diagnostic features from the Breast Cancer Wisconsin dataset.

The application combines a trained Logistic Regression model with a Flask backend and an interactive web-based user interface.

---

## Features

* Real-time breast cancer prediction
* Interactive and user-friendly interface
* Flask-based REST API
* Machine Learning model integration
* Confidence score display
* End-to-end ML deployment workflow

---

## Technology Stack

### Machine Learning

* Python
* Scikit-learn
* NumPy
* Pandas

### Backend

* Flask

### Frontend

* HTML
* CSS
* JavaScript

### Model Storage

* Pickle

---

## Project Structure

```text
Breast Cancer Detection Model/
│
├── app.py
├── train.py
├── model.pkl
├── requirements.txt
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

---

## Model Performance

| Metric    | Score   |
| --------- | ------- |
| Accuracy  | 97.37%  |
| Precision | 95.95%  |
| Recall    | 100.00% |
| F1 Score  | 97.96%  |

### Key Observation

Recall is particularly important in medical diagnosis because it measures the model's ability to correctly identify malignant cases. A recall of 100% on the test set indicates that no malignant cases were missed during evaluation.

---

## Workflow

### 1. Model Training

* Load and preprocess dataset
* Split data into training and testing sets
* Train Logistic Regression model
* Evaluate performance
* Save trained model as `model.pkl`

### 2. Backend Development

* Build Flask server
* Load trained model
* Create `/predict` endpoint
* Process incoming data
* Return prediction and confidence score

### 3. Frontend Development

* Create input forms using HTML
* Style application using CSS
* Use JavaScript Fetch API for communication
* Display predictions dynamically

### 4. Integration

* User enters diagnostic values
* Frontend sends data to Flask API
* Model generates prediction
* Results are returned and displayed instantly

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the Flask Application

```bash
python app.py
```

### Open Browser

```text
http://127.0.0.1:5000
```

---

## Future Improvements

* Deploy application to cloud platforms
* Add model explainability using SHAP
* Improve UI responsiveness
* Support multiple machine learning models
* Add patient report generation

---

## Author

Shruti Singh

AI & Machine Learning Student

End-to-End Machine Learning Project using Flask and Scikit-learn.
