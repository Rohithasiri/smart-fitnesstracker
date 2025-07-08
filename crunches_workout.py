import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load crunches video (change to your file if needed)
cap = cv2.VideoCapture(r"C:\Users\O Lakshmi reddy\Downloads\crunches 1.mp4")  # Replace with your video file

# Rep counter variables
counter = 0
stage = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Use right side landmarks
        shoulder_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h
        hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h

        # Debug Y positions
        cv2.putText(image, f'ShY: {int(shoulder_y)}  HipY: {int(hip_y)}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Crunch detection logic based on shoulder Y motion
        if shoulder_y > hip_y + 25:
            stage = "down"
        elif shoulder_y < hip_y + 5 and stage == "down":
            stage = "up"
            counter += 1

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Overlay counter only
    cv2.rectangle(image, (0, 0), (300, 80), (0, 0, 0), -1)
    cv2.putText(image, f'Crunches: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Resize frame for full visibility
    scaled_frame = cv2.resize(image, (960, 540))  # Use (1280, 720) if you want bigger
    cv2.imshow("Crunches Tracker", scaled_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
