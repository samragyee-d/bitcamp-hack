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
import os

#Import Environment Variables
load_dotenv()

MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY')

# Load Haar Cascade + emotion detection model
face_classifier = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_detection_model_100epochs.h5')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


app = Flask(__name__)

# Load YOLOv5 model from torch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # confidence threshold
'''
pip install flask opencv-python torch torchvision
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
'''
# List of phone-like classes to detect (YOLOv5 doesn't explicitly have "phone")
PHONE_CLASSES = ['cell phone']

# Load Haar cascade for face detection
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# More SQL setup
app.secret_key = os.getenv('FLASK_SECRET_KEY')
mysql_password = os.getenv('MYSQL_PASSWORD')


def generate_frames():
    '''while True:
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
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')'''
    '''cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB for YOLO
        results = model(frame[..., ::-1])  # BGR to RGB

        for det in results.xyxy[0]:
            xmin, ymin, xmax, ymax, conf, cls = det
            label = results.names[int(cls)]
            if label in PHONE_CLASSES:
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')'''
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Face detection using Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Phone detection using YOLOv5
        results = model(frame[..., ::-1])  # BGR to RGB

        for det in results.xyxy[0]:
            xmin, ymin, xmax, ymax, conf, cls = det
            label = results.names[int(cls)]
            if label in PHONE_CLASSES:
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Return frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')  # Added for email
        password = request.form.get('password')

        if not username or not email or not password:
            return render_template('register.html', message='Please fill out all fields.')

        try:
            connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password=os.getenv('MYSQL_PASSWORD'),
                database=os.getenv('MYSQL_DATABASE')
            )
            
            cursor = connection.cursor()
            insert_query = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
            cursor.execute(insert_query, (username, email, password))
            connection.commit()

            cursor.close()
            connection.close()

            return render_template('registration.html', message='Registration successful!')

        except mysql.connector.Error as err:
            return render_template('registration.html', message=f"Error: {err}")

    return render_template('registration.html')  # Changed to render 'register.html' for GET method
print("MYSQL_PASSWORD:", os.getenv('MYSQL_PASSWORD'))
print("MYSQL_DATABASE:", os.getenv('MYSQL_DATABASE'))


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

        if result and result[0] == password:
            return "Login successful!"
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
