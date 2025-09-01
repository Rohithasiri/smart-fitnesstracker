import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st

def run_crunches(user_weight, target_reps=10, video_path=0):
    # MET value for crunches (approx.)
    MET = 4.0

    # MediaPipe setup
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    crunch_count = 0
    position = None
    feedback_text = ""
    start_time = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Unable to open video source.")
        return {"exercise": "Crunches", "reps": 0, "calories": 0, "status": "Fail"}


    frame_placeholder = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if start_time is None:
                start_time = time.time()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Key points for crunch detection
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                angle = calculate_angle(shoulder, hip, knee)

                # Feedback
                if angle >120:  # crunch down
                    position = "down"
                    feedback_text="Go Down"
                if angle < 110 and position == "down":  # crunch up
                    position = "up"
                    crunch_count += 1
                    feedback_text = f"Crunch #{crunch_count}"
                    feedback_text = "Good Job!"

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except Exception:
                feedback_text = "Pose not detected"

            # Display counter and feedback
            cv2.rectangle(image, (0, 0), (500, 100), (0,0,0), -1)
            cv2.putText(image, 'CRUNCHES', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f'Count:{(crunch_count)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, feedback_text, (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            frame_placeholder.image(image, channels='BGR')
            
            if crunch_count >= target_reps:
                        break
    cap.release()

    # Calories burned calculation
    elapsed_time = time.time() - start_time if start_time else 0
    calories_burned = MET * user_weight * (elapsed_time / 3600)

    return {
        "exercise": "CRUNCHES",
        "reps": crunch_count,
        "duration_sec": int(elapsed_time),
        "calories": round(calories_burned, 2),
        "status": "Success" if crunch_count >= target_reps else "Fail",
        "success": crunch_count>= target_reps
    }
