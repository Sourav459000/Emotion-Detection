import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('emotion_recognition_model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam
cap = cv2.VideoCapture("rtsp://admin:abcd1234@172.22.57.25")

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Preprocessing function for frames
def preprocess_frame(frame, target_size=(48, 48)):
    resized_frame = cv2.resize(frame, target_size)
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
    return reshaped_frame

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale for FER2013 dataset compatibility
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colored_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel image

    # Preprocess the frame
    processed_frame = preprocess_frame(colored_frame)

    # Predict emotion
    predictions = model.predict(processed_frame)
    emotion_index = np.argmax(predictions)
    emotion_label = emotion_labels[emotion_index]

    # Display the emotion on the frame
    cv2.putText(frame, emotion_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Real-Time Emotion Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()