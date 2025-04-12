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
            # Optional: resize the frame or do processing here
            # frame = cv2.resize(frame, (640, 480))

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame in multipart format
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
