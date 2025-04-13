import os
import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from gemini import generate_gemini_response
import time
import requests
from state import chat_history


load_dotenv()

# Load Haar cascade for face detection
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Emotion labels
emotion_labels = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise'
}

# Load emotion model
# FOR ALVIA THE ROUTE IS bitcamp-hack/models/facialemotionmodel.h5
# FOR EVERYONE ELSE IT IS models/facialemotionmodel.h5
emotion_model = load_model(
    "bitcamp-hack/models/facialemotionmodel.h5"
)
# Load YOLOv5 model (nano version for performance)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5
if torch.cuda.is_available():
    model.to('cuda')

# Helper function for emotion input
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Phone detection variables
phone_detected_start = None
phone_alert_sent = False
phone_detection_threshold = 1  # seconds

from collections import deque

# Track recent emotions (rolling window)
emotion_history = deque(maxlen=20)  # adjust size as needed
negative_emotions = {'angry', 'disgust', 'fear'}
emotion_alert_sent = False

from datetime import datetime, timedelta

# Track timestamps of comforting messages
comforting_message_times = deque()
comforting_message_limit = 1  # threshold
comforting_message_window_minutes = 1  # x minutes
break_alert_sent = False
chat_history = []

# Cooldown state
last_phone_message_time = time.time()
last_emotion_message_time = time.time()

phone_cooldown = 0.5     # seconds
emotion_cooldown = 2  # seconds

def generate_frames():
    cap = cv2.VideoCapture(0)

    global phone_detected_start, phone_alert_sent, emotion_history, negative_emotions, emotion_alert_sent, comforting_message_times, comforting_message_limit, comforting_message_window_minutes, break_alert_sent, chat_history, last_phone_message_time, last_emotion_message_time, phone_cooldown, emotion_cooldown

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Step 1: Run YOLO for phone detection
        results = model(frame)
        labels = results.names
        phone_detected_this_frame = False

        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.5:
                label = labels[int(cls)]
                if label == 'cell phone':
                    phone_detected_this_frame = True
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Cell Phone: {conf*100:.2f}%', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Handle timing logic
        if phone_detected_this_frame:
            if phone_detected_start is None:
                phone_detected_start = time.time()
            elif not phone_alert_sent and time.time() - phone_detected_start >= phone_detection_threshold:
                if time.time() - last_phone_message_time > phone_cooldown:
                    response = generate_gemini_response("Send a message scolding the user for having a phone present.")
                    phone_alert_sent = True
                    last_phone_message_time = time.time()
                    requests.post("http://127.0.0.1:5000/push_system_message", json={"message": response})

        else:
            # Reset if phone not detected
            phone_detected_start = None
            phone_alert_sent = False

        # Step 2: Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                                scaleFactor=1.1,     # Reduce this slightly for stricter scale
                                                minNeighbors=6,      # Increase this for stricter grouping
                                                minSize=(60, 60)     # Set a minimum face size
                                            )


        # Step 3: If faces exist, run emotion detection
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (48, 48))
                img = extract_features(face_resized)
                pred = emotion_model.predict(img, verbose=0)
                # Penalize 'sad' slightly
                adjusted_pred = pred[0]
                adjusted_pred[5] *= 0.85  # Lower 'sad' confidence artificially
                prediction_label = emotion_labels[np.argmax(adjusted_pred)]
                confidence = np.max(pred)

                # Add to emotion history
                emotion_history.append(prediction_label)

                # Count recent negative emotions
                negative_count = sum(1 for e in emotion_history if e in negative_emotions)

                # Trigger response if threshold is exceeded
                if negative_count >= 7:
                    if not emotion_alert_sent and time.time() - last_emotion_message_time > emotion_cooldown:
                        message = generate_gemini_response("The user seems emotionally distressed. Please send a short comforting or encouraging message.")
                        requests.post("http://127.0.0.1:5000/push_system_message", json={"message": message})
                        emotion_alert_sent = True
                        last_emotion_message_time = time.time()
                        comforting_message_times.append(datetime.now())

                else:
                    emotion_alert_sent = False

                # Clean up old timestamps outside the window
                now = datetime.now()
                comforting_message_times = deque(
                    t for t in comforting_message_times if now - t <= timedelta(minutes=comforting_message_window_minutes)
                )

                # If too many comforting messages recently, suggest a break
                if len(comforting_message_times) > comforting_message_limit and not break_alert_sent:
                    message = generate_gemini_response("The user has received multiple comforting messages recently. Recommend taking a short break to rest and reset.")
                    requests.post("http://127.0.0.1:5000/push_system_message", json={"message": message})
                    break_alert_sent = True
                elif len(comforting_message_times) <= comforting_message_limit:
                    break_alert_sent = False  # Reset once count drops


                cv2.putText(frame, f'{prediction_label}: {confidence*100:.2f}%', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Return frame as byte stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
