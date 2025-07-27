import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st

def run_squats(user_weight, target_reps=10, video_path=0):
    MET = 5.0  # MET value for squats
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    squat_count = 0
    squat_position = None
    feedback_text = ""
    start_time = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Unable to open video source.")
        return {"exercise": "Squats", "reps": 0, "calories": 0, "status": "Fail"}

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
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)
                h, w, _ = image.shape
                knee_point = tuple(np.multiply(knee, [w, h]).astype(int))

                # Display angle
                cv2.putText(image, str(int(angle)), (knee_point[0] + 10, knee_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.ellipse(image, knee_point, (40, 40), 0, 0, int(angle), (0, 255, 0), 2)

                # Squat logic
                if angle < 70:
                    if squat_position != 'down':
                        squat_position = 'down'
                        feedback_text = "Go deeper!" if angle > 60 else "Good squat!"
                elif angle > 160:
                    if squat_position == 'down':
                        squat_position = 'up'
                        squat_count += 1
                        feedback_text = "Nice! Stand complete."
                        if squat_count >= target_reps:
                            break
                else:
                    feedback_text = "Stand up straight!"

            except Exception:
                feedback_text = "Pose not detected"

            # UI Overlay
            cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
            cv2.putText(image, 'SQUATS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(image, str(squat_count), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            cv2.putText(image, feedback_text, (280, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            frame_placeholder.image(image, channels='BGR')

    cap.release()

    elapsed_time = time.time() - start_time if start_time else 0
    calories_burned = MET * user_weight * (elapsed_time / 3600)

    return {
        "exercise": "Squats",
        "reps": squat_count,
        "duration_sec": int(elapsed_time),
        "calories": round(calories_burned, 2),
        "status": "Success" if squat_count >= target_reps else "Fail",
        "success": squat_count >= target_reps
    }
