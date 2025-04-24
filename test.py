import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("drowsiness_detector_model.h5")

image_path = "dataset/awake/s0037_09977_1_1_1_0_0_01.png"
image = cv2.imread(image_path)


image = cv2.resize(image, (80, 80))

image = image.astype("float32") / 255.0

image = np.expand_dims(image, axis=0) 

pred = model.predict(image)[0][0]
print(f"Предсказание: {pred}")

if pred > 0.5:
    print("Глаза открыты (Awake)")
else:
    print("Глаза закрыты (Asleep)")