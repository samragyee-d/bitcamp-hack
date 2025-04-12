# app.py

from flask import Flask, render_template, request, redirect, url_for, Response, flash
import cv2
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

#SQL setup
import mysql.connector
from dotenv import load_dotenv
import os
from tensorflow.keras.models import model_from_json

# Import Environment Variables
load_dotenv()

app = Flask(__name__)
import cv2
import torch

# Load YOLOv5 model from torch hub for object detection (including cell phones)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # confidence threshold

# Load the Emotion Recognition Model
json_file = open("/Users/alvianaqvi/vsc_code/bitcamp-hack-alvia/Emotion-recognition/models/facial-emotion-recognition-higher-accuracy/facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
emotion_model = load_model("/Users/alvianaqvi/vsc_code/bitcamp-hack-alvia/Emotion-recognition/models/facial-emotion-recognition-higher-accuracy/facialemotionmodel.h5")

# Emotion labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Load Haar cascade for face detection
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Helper function for emotion extraction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Face detection using Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract face for emotion recognition
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (48, 48))
            img = extract_features(face_resized)
            pred = emotion_model.predict(img)
            prediction_label = emotion_labels[pred.argmax()]
            confidence = np.max(pred)

            # Display the predicted emotion and confidence
            cv2.putText(frame, f'{prediction_label}: {confidence*100:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Object detection using YOLOv5 (detecting cell phones and other objects)
        results = model(frame)
        labels = results.names  # Get class labels
        for *xyxy, conf, cls in results.xyxy[0]:  # loop over detections
            if conf > 0.5:  # confidence threshold
                label = labels[int(cls)]  # Get the detected object class label
                if label == 'cell phone':  # Check if the detected object is a cell phone
                    x1, y1, x2, y2 = map(int, xyxy)  # Get bounding box coordinates
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box for cell phone
                    cv2.putText(frame, f'Cell Phone: {conf*100:.2f}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Return frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return render_template('video.html')  # HTML with <img src="/video_feed">


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_input = request.form['username']
        password = request.form['password']

        # Connect to MySQL
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        cursor = conn.cursor()

        # Fetch the stored hashed password from the database
        cursor.execute("SELECT password FROM users WHERE username=%s OR email=%s", (user_input, user_input))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        # Check if user exists and if the password matches
        if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
            return video()
        else:
            flash("Invalid credentials. Please try again.")
            return redirect(url_for('login'))

    return render_template('Login.html')

@app.route('/video')
def video():
    return render_template('video.html')  # HTML with <img src="/video_feed">

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# BASIC FLASK
@app.route('/pagename')
def pagename():
    return render_template('pagename.html')

'''
@app.route('/submit', methods=['POST'])
def submit():
    # Get form data
    name = request.form['name']
    email = request.form['email']
    
    # You can process or store the data here
    return render_template('result.html', name=name, email=email)
'''

#alvia naqvi

if __name__ == '__main__':
    app.run(debug=True)
