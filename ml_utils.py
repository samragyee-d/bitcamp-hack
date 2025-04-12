import os
import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from gemini import generate_gemini_response
import time

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
phone_detection_threshold = 5  # seconds

def generate_frames():
    cap = cv2.VideoCapture(0)

    global phone_detected_start, phone_alert_sent

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
                print(generate_gemini_response("Send a message scolding the user for having a phone present."))
                phone_alert_sent = True
        else:
            # Reset if phone not detected
            phone_detected_start = None
            phone_alert_sent = False

        # Step 2: Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Step 3: If faces exist, run emotion detection
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (48, 48))
                img = extract_features(face_resized)
                pred = emotion_model.predict(img, verbose=0)
                prediction_label = emotion_labels[pred.argmax()]
                confidence = np.max(pred)

                cv2.putText(frame, f'{prediction_label}: {confidence*100:.2f}%', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Return frame as byte stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
