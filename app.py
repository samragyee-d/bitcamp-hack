# app.py

from flask import Flask, render_template, request, redirect, url_for, Response, flash
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

#SQL setup
import mysql.connector
from dotenv import load_dotenv
from ml_utils import generate_frames
import os

#Import Environment Variables
load_dotenv()

app = Flask(__name__)
import cv2
import torch

app = Flask(__name__)

# Load YOLOv5 model from torch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # confidence threshold
'''
pip install flask opencv-python torch torchvision
git clone https://github.com/ultralytics/yolov5  # If using YOLOv5
cd yolov5
pip install -r requirements.txt
'''
# List of phone-like classes to detect (YOLOv5 doesn't explicitly have "phone")
PHONE_CLASSES = ['cell phone']

# More SQL setup
app.secret_key = os.getenv('FLASK_SECRET_KEY')
mysql_password = os.getenv('MYSQL_PASSWORD')

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
def video():
    return render_template('video.html')  # HTML with <img src="/video_feed">

@app.route('/backendvideo')
def backendvideo():
    return render_template('backendvideo.html')  # HTML with <img src="/video_feed">

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