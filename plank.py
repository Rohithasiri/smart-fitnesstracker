import cv2
import time
import mediapipe as mp
import numpy as np
import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

CRITICAL_JOINTS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

def check_plank_posture(landmarks):
    def get_point(lm): return np.array([lm.x, lm.y])
    left_shoulder = get_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    left_hip = get_point(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    left_ankle = get_point(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    v1 = left_shoulder - left_hip
    v2 = left_ankle - left_hip
    angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

    return 160 < angle < 180  # Ideal plank angle

def run_plank(user_weight, target_time=30, video_path=0):
    MET = 3.0
    visibility_threshold = 0.5
    sets = 0
    in_plank = False
    plank_start_time = None
    elapsed = 0.0
    detected_once = False  # ✅ to check if posture was ever detected

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video source")
        return {"exercise": "Plank", "duration_sec": 0, "sets": 0, "calories": 0, "status": "Not Detected"}

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    frame_placeholder = st.empty()
    start_time = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_time is None:
            start_time = time.time()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # ✅ Handle pause/resume/exit
        if st.session_state.workout_status == "exit":
            break
        if st.session_state.workout_status == "paused":
            if not st.session_state.pause_message_shown:
                st.warning("⏸ Workout Paused. Press Resume to continue.")
                st.session_state.pause_message_shown = True
            time.sleep(1)
            continue
        else:
            st.session_state.pause_message_shown = False

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            joints_visible = all(
                landmarks[j.value].visibility > visibility_threshold
                for j in CRITICAL_JOINTS
            )

            if joints_visible:
                posture_ok = check_plank_posture(landmarks)

                if posture_ok:
                    detected_once = True  # ✅ posture detected at least once
                    if not in_plank:
                        plank_start_time = time.time()
                        in_plank = True
                    elapsed = time.time() - plank_start_time
                    if elapsed >= target_time:
                        break
                else:
                    if in_plank:
                        sets += 1
                        in_plank = False
                        plank_start_time = None
            else:
                if in_plank:
                    sets += 1
                    in_plank = False
                    plank_start_time = None

            # Overlay info
            cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frame_placeholder.image(frame, channels="BGR", caption="Plank Tracker")

    cap.release()

    duration_hr = elapsed / 3600
    calories_burned = MET * user_weight * duration_hr

    # ✅ Better status logic
    if not detected_once:
        status = "Not Detected"
    elif elapsed >= target_time:
        status = "Success"
    else:
        status = "Incomplete"

    return {
        "exercise": "Plank",
        "duration_sec": round(elapsed, 2),
        "sets": sets,
        "calories": round(calories_burned, 2),
        "status": status
    }
