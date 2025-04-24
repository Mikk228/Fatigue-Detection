import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model_epoch_10.keras")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

last_eye_roi = None
last_eye_coords = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    eyes_detected = False

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
            last_eye_roi = eye_roi
            last_eye_coords = (x+ex, y+ey, ew, eh)
            eyes_detected = True

    if not eyes_detected and last_eye_roi is not None and last_eye_coords is not None:
        eye_roi = last_eye_roi
        x, y, ew, eh = last_eye_coords

    elif eyes_detected:
        last_eye_roi = eye_roi
        last_eye_coords = (x+ex, y+ey, ew, eh)

    if last_eye_roi is not None:
        resized = cv2.resize(last_eye_roi, (80, 80))
        normalized = resized / 255.0
        input_image = np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)

        prediction = model.predict(input_image)[0][0]
        status = f"Asleep {prediction}" if prediction < 0.5 else f"Awake {prediction}"


        cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Driver State", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
