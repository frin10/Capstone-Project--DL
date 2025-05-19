# realtime_predict.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import string

model = load_model("asl_model.h5")
classes = list(string.ascii_uppercase)

def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI)
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    img = preprocess(roi)
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    class_label = classes[class_id]

    # Display prediction
    cv2.putText(frame, f"Prediction: {class_label}", (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("ASL Real-time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    

cap.release()
cv2.destroyAllWindows()