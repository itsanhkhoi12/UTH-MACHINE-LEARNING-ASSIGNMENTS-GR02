from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import joblib
from pydantic import BaseModel
import scipy.sparse as sp
import os

# Đường dẫn cũ của bạn
MODEL_DIR = r'C:\University\May-hoc\Lab\UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02\practice_1\models'
DATA_DIR = r'C:\University\May-hoc\Lab\UTH-MACHINE-LEARNING-ASSIGNMENTS-GR02\practice_1\data\ready_for_train'

model = joblib.load(os.path.join(MODEL_DIR, 'Logistic_Regression.pkl'))
tfidf_subject = joblib.load(os.path.join(DATA_DIR, 'tfidf_subject.pkl'))
tfidf_message = joblib.load(os.path.join(DATA_DIR, 'tfidf_message.pkl'))

app = FastAPI()

# Giao diện HTML đơn giản
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Spam Detector</title></head>
<body>
    <h2>Phân loại Email Spam</h2>
    <form action="/predict" method="post">
        Subject: <input type="text" name="subject" id="subject"><br><br>
        Message: <textarea name="message" id="message"></textarea><br><br>
        <button onclick="predict()">Dự đoán</button>
    </form>
    <h3 id="result"></h3>
    <script>
    async function predict() {
        event.preventDefault();
        const data = { 
            subject: document.getElementById('subject').value, 
            message: document.getElementById('message').value 
        };
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const res = await response.json();
        document.getElementById('result').innerText = "Kết quả: " + res.prediction;
    }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def get_home():
    return HTML_TEMPLATE

@app.post("/predict")
def predict_email(data: dict):
    subj_vec = tfidf_subject.transform([data['subject']])
    msg_vec = tfidf_message.transform([data['message']])
    combined_features = sp.hstack([subj_vec, msg_vec])
    prediction = model.predict(combined_features)
    result = "Spam" if prediction[0] == 1 else "Ham"
    return {"prediction": result}