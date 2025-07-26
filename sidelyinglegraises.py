import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

MET = 3.5  # MET value for side-lying leg raises

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180 / np.pi)
    return 360 - angle if angle > 180 else angle

def run_sidelying_leg_raises(user_weight, target_reps=10, video_path=0):
    count = 0
    direction = 0  # 0 = down, 1 = up
    threshold_up = 0.40
    threshold_down = 0.55

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video source")
        return {"exercise": "Side-Lying Leg Raises", "reps": 0, "calories": 0, "status": "Fail"}

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    start_time = None
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_time is None:
            start_time = time.time()

        frame = cv2.flip(frame, 1)  # mirror for webcam
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)
            ankle_y = ankle[1]

            if ankle_y < threshold_up and direction == 0:
                direction = 1
            elif ankle_y > threshold_down and direction == 1:
                count += 1
                if count >= target_reps:
                    break
                direction = 0

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame, f'Angle: {int(angle)}°', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, f'Reps: {count}', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        frame_placeholder.image(frame, channels="BGR", caption="Side-Lying Leg Raise Tracker")

    cap.release()

    elapsed_time = time.time() - start_time if start_time else 0
    calories_burned = MET * user_weight * (elapsed_time / 3600)

    return {
        "exercise": "Side-Lying Leg Raises",
        "reps": count,
        "duration_sec": int(elapsed_time),
        "calories": round(calories_burned, 2),
        "status": "Success" if count >= target_reps else "Fail",
        "success": count >= target_reps
    }
