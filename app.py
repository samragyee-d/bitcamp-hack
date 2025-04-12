# app.py

from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import google.generativeai as genai

# Load Haar Cascade + emotion detection model
face_classifier = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_detection_model_100epochs.h5')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


app = Flask(__name__)
camera = cv2.VideoCapture(0)  # 0 = default webcam

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
