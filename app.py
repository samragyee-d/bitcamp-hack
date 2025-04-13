# app.py

from flask import Flask, render_template, request, redirect, url_for, Response, flash
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import requests
from datetime import datetime


#SQL setup
import mysql.connector
from dotenv import load_dotenv
from ml_utils import generate_frames
import bcrypt
import os
from state import clear_chat_history, recording_flag


#Import Environment Variables
load_dotenv()

from queue import Queue

system_message_queue = Queue()

app = Flask(__name__)
import cv2
import torch

app = Flask(__name__)

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
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/logout')
def logout(): 
    session.clear() 
    return redirect(url_for('home', logout='true'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            return render_template('registration.html', message='Please fill out all fields.')

        try:
            # Hash the password using bcrypt
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            # Connect to MySQL
            connection = mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE
            )
            cursor = connection.cursor()

            # Insert the user into the database
            insert_query = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
            cursor.execute(insert_query, (username, email, hashed_password))
            connection.commit()

            cursor.close()
            connection.close()

            # Redirect to homepage with success flag
            return redirect(url_for('home', registered='true'))

        except mysql.connector.Error as err:
            print(f"Error during insert: {err}")
            return render_template('registration.html', message=f"Error: {err}")

    return render_template('registration.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_input = request.form['username']
        password = request.form['password']

        # Connect to MySQL
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        cursor = connection.cursor()
        cursor.execute("SELECT password FROM users WHERE username=%s OR email=%s", (user_input, user_input))
        result = cursor.fetchone()
        cursor.close()
        connection.close()

        if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
            return redirect(url_for('home', login = 'true'))
        else:
            flash("Invalid credentials. Please try again.")
            return redirect(url_for('login'))

    return render_template('Login.html')


@app.route('/video')
def video():
    clear_chat_history()
    return render_template('video.html')  # HTML with <img src="/video_feed">

@app.route('/backendvideo')
def backendvideo():
    global recording
    # --- Send welcome message on startup ---
    welcome_message = generate_gemini_response("Welcome the user to their study session with a friendly and motivating tone.")
    requests.post("http://127.0.0.1:5000/push_system_message", json={"message": welcome_message})
    return render_template('backendvideo.html', is_recording=recording_flag["status"])


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

from gemini import generate_gemini_response
from flask import jsonify

@app.route('/gemini_chat', methods=['POST'])
def gemini_chat():
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({'response': 'No message provided.'})
    
    response = generate_gemini_response(message)
    return jsonify({'response': response})

@app.route('/system_chat', methods=['GET'])
def system_chat():
    if not system_message_queue.empty():
        message = system_message_queue.get()
        return jsonify({'response': message})
    else:
        return jsonify({'response': None})

@app.route('/push_system_message', methods=['POST'])
def push_system_message():
    data = request.get_json()
    message = data.get('message', '')
    if message:
        system_message_queue.put(message)
    return jsonify({'status': 'success'})


from flask import session

@app.route('/start_recording', methods=['POST'])
def start_recording():
    recording_flag["status"] = True
    return "Recording started"

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    recording_flag["status"] = False
    return "Recording stopped"

@app.route('/download')
def download_video():
    return redirect(url_for('static', filename='recorded.mp4'))

@app.route('/recording_status')
def recording_status():
    return jsonify({"status": recording_flag["status"]})

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

if __name__ == '__main__':
    app.run(debug=True)