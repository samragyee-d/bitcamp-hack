# app.py

from flask import Flask, render_template, request, redirect, url_for, Response, flash
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import google.generativeai as genai
from werkzeug.security import generate_password_hash, check_password_hash



#SQL setup
import mysql.connector
from dotenv import load_dotenv
import os

#Import Environment Variables
load_dotenv()

# Load Haar Cascade + emotion detection model
face_classifier = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_detection_model_100epochs.h5')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


app = Flask(__name__)
camera = cv2.VideoCapture(0)  # 0 = default webcam

# More SQL setup
app.secret_key = os.getenv('FLASK_SECRET_KEY')
mysql_password = os.getenv('MYSQL_PASSWORD')


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # Draw face box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Extract face ROI and preprocess
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Predict emotion
                preds = emotion_model.predict(roi)[0]
                label = class_labels[preds.argmax()]
                label_position = (x, y - 10)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert frame to bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield as video stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_input = request.form['username']
        password = request.form['password']

        # Connect to MySQL only when needed
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password=os.getenv('MYSQL_PASSWORD'),
            database="flask_auth"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username=%s OR email=%s", (user_input, user_input))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result and check_password_hash(result[0], password):
            return "Login successful!"
        else:
            flash("Invalid credentials. Please try again.")
            return redirect(url_for('login'))

    return render_template('Login.html')


#Register User

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_pw = generate_password_hash(password)

        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                       (username, email, hashed_pw))
        conn.commit()

        flash("Registration successful! Please log in.")
        return redirect(url_for('login'))

    return render_template('register.html')  # You'll need to create this HTML

@app.route('/video')
def vdeo():
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
