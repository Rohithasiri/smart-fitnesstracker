# 

# 

# 

# 

# 

import cv2
import mediapipe as mp
import numpy as np

# Setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load video
video_path = "/Users/rohithayenugu/Downloads/WhatsApp Video 2025-06-09 at 22.03.59.mp4"
cap = cv2.VideoCapture(video_path)

# Rep tracking
count = 0
direction = 0  # 0 = down, 1 = up
prev_y = None
threshold_up = 0.40  # leg raised
threshold_down = 0.55  # leg lowered

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (720, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        ankle_y = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y  # Normalized y-position

        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display current ankle y-position
        cv2.putText(frame, f"Ankle Y: {ankle_y:.2f}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Rep logic based on vertical movement (y axis)
        if ankle_y < threshold_up:  # leg is UP
            if direction == 0:
                direction = 1
        elif ankle_y > threshold_down:  # leg is DOWN
            if direction == 1:
                count += 1
                direction = 0

        # Draw rep count
        cv2.putText(frame, f"Reps: {count}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Side-Lying Leg Raise (Vertical Tracker)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
