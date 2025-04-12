# ml_utils.py

import os
import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()

# Load Haar cascade
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Emotion labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Load emotion model
emotion_model = load_model("/Users/alvianaqvi/vsc_code/bitcamp-hack-alvia/Emotion-recognition/models/facial-emotion-recognition-higher-accuracy/facialemotionmodel.h5")

# Load YOLOv5 nano model (smaller = faster)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.5
if torch.cuda.is_available():
    model.to('cuda')

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