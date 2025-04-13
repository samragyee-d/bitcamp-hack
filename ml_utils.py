import os
import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from gemini import generate_gemini_response
import time
from datetime import datetime
import requests
from state import chat_history, recording_flag
import mysql.connector
from collections import deque

load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST'),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': os.getenv('MYSQL_DATABASE')
}

# Ensure the 'static' folder exists
STATIC_FOLDER = 'static/'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

def get_db_connection():
    """Create and return a new database connection"""
    return mysql.connector.connect(**DB_CONFIG)

def save_video_to_db(user_id, video_path):
    """Save video metadata to database"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO user_videos (user_id, video_path) VALUES (%s, %s)",
            (user_id, os.path.basename(video_path))  # Store only filename
        )
        connection.commit()
        return True
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return False
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

# Path for saving the video file
def get_video_path():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(STATIC_FOLDER, f"recording_{timestamp}.mp4")

# Load Haar cascade for face detection
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Emotion labels
emotion_labels = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise'
}

# Load emotion model
emotion_model = load_model("models/facialemotionmodel.h5")

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

# Track recent emotions (rolling window)
emotion_history = deque(maxlen=20)
negative_emotions = {'angry', 'disgust', 'fear'}
emotion_alert_sent = False

# Track timestamps of comforting messages
comforting_message_times = deque()
comforting_message_limit = 1
comforting_message_window_minutes = 1
break_alert_sent = False

# Cooldown state
last_phone_message_time = time.time()
last_emotion_message_time = time.time()
phone_cooldown = 1     # seconds
emotion_cooldown = 2  # seconds

video_writer = None
recorded_frames = []
frame_size = (640, 480)
fps = 20  # Normal capture rate

def generate_frames():
    global video_writer, recorded_frames
    
    # Get user_id from recording_flag (passed from Flask)
    user_id = recording_flag.get("user_id")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    while True:
        success, frame = cap.read()
        if not success:
            break

        global phone_detected_start, phone_alert_sent, emotion_history, emotion_alert_sent
        global comforting_message_times, break_alert_sent, last_phone_message_time, last_emotion_message_time

        # Phone detection logic
        results = model(frame)
        labels = results.names
        phone_detected_this_frame = False

        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.5 and labels[int(cls)] == 'cell phone':
                phone_detected_this_frame = True
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Cell Phone: {conf*100:.2f}%', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Phone timing logic
        if phone_detected_this_frame:
            if phone_detected_start is None:
                phone_detected_start = time.time()
            elif not phone_alert_sent and time.time() - phone_detected_start >= phone_detection_threshold:
                if time.time() - last_phone_message_time > phone_cooldown:
                    response = generate_gemini_response("Send a message telling the user to put down their phone and get back to studying.")
                    phone_alert_sent = True
                    last_phone_message_time = time.time()
                    requests.post("http://127.0.0.1:5000/push_system_message", json={"message": response})
        else:
            phone_detected_start = None
            phone_alert_sent = False

        # Face and emotion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (48, 48))
                img = extract_features(face_resized)
                pred = emotion_model.predict(img, verbose=0)
                adjusted_pred = pred[0]
                adjusted_pred[5] *= 0.85  # Lower 'sad' confidence
                prediction_label = emotion_labels[np.argmax(adjusted_pred)]
                confidence = np.max(pred)

                # Emotion tracking
                emotion_history.append(prediction_label)
                negative_count = sum(1 for e in emotion_history if e in negative_emotions)

                if negative_count >= 7 and not emotion_alert_sent and time.time() - last_emotion_message_time > emotion_cooldown:
                    message = generate_gemini_response("The user has had negative facial expressions recently. Send an encouraging message.")
                    requests.post("http://127.0.0.1:5000/push_system_message", json={"message": message})
                    emotion_alert_sent = True
                    last_emotion_message_time = time.time()
                    comforting_message_times.append(datetime.now())
                elif negative_count < 7:
                    emotion_alert_sent = False

                # Clean up old timestamps
                now = datetime.now()
                comforting_message_times = deque(
                    t for t in comforting_message_times if now - t <= timedelta(minutes=comforting_message_window_minutes)
                )

                if len(comforting_message_times) > comforting_message_limit and not break_alert_sent:
                    message = generate_gemini_response("The user has received multiple comforting messages recently. Recommend taking a short break.")
                    requests.post("http://127.0.0.1:5000/push_system_message", json={"message": message})
                    break_alert_sent = True
                elif len(comforting_message_times) <= comforting_message_limit:
                    break_alert_sent = False

                cv2.putText(frame, f'{prediction_label}: {confidence*100:.2f}%', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Recording logic
        if recording_flag["status"]:
            recorded_frames.append(frame.copy())
        elif recorded_frames:  # Recording just stopped
            # Save sped-up video
            sped_up_fps = fps * 4
            video_path = get_video_path()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, sped_up_fps, frame_size)

            for f in recorded_frames[::4]:  # Keep every 4th frame
                resized = cv2.resize(f, frame_size)
                video_writer.write(resized)
            
            video_writer.release()
            
            # Save to database if user is logged in
            if user_id:
                save_video_to_db(user_id, video_path)
            
            recorded_frames.clear()

        # Stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')