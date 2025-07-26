import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow/MediaPipe logs

MET = 8.0  # MET value for moderate-intensity pushups

def calculate_angle(a, b, c):
    angle = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]) -
                       np.arctan2(a[1] - b[1], a[0] - b[0]))
    return abs(angle)

def run_pushups(user_weight, target_reps=10, video_path=0):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video source")
        return {"exercise": "Push-ups", "reps": 0, "calories": 0, "status": "Fail"}

    reps = 0
    up_position = False
    down_position = False
    start_time = None
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_time is None:
            start_time = time.time()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            if angle > 170:
                if down_position:
                    reps += 1
                    if reps >= target_reps:
                        break
                up_position = True
                down_position = False
            elif 70 < angle < 100:
                down_position = True
                up_position = False

            cv2.putText(image, f"Push-ups: {reps}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        frame_placeholder.image(image, channels='BGR', caption='Push-up Tracker')

    cap.release()

    elapsed_sec = time.time() - start_time if start_time else 0
    calories = MET * user_weight * (elapsed_sec / 3600)

    return {
        "exercise": "Push-ups",
        "reps": reps,
        "duration_sec": round(elapsed_sec, 2),
        "calories": round(calories, 2),
        "status": "Success" if reps >= target_reps else "Fail"
    }
