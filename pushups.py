import cv2
import mediapipe as mp
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow/MediaPipe logs
# Constants
MET = 8.0  # MET value for moderate-intensity pushups

def calculate_angle(a, b, c):
    angle = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]) -
                       np.arctan2(a[1] - b[1], a[0] - b[0]))
    return abs(angle)

def run_pushups(user_weight, target_reps=10, video_path="Sample Videos/pushups.mp4"):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    reps = 0
    up_position = False
    down_position = False
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            if angle > 170:
                if down_position:
                    reps += 1
                    print(f"Push-ups: {reps}")
                    if(reps>=target_reps):
                        break
                up_position = True
                down_position = False
            elif 70 < angle < 100:
                down_position = True
                up_position = False

        cv2.putText(image, f"Push-ups: {reps}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow("Push-up Tracker", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    elapsed_sec = time.time() - start_time
    calories = MET * user_weight * (elapsed_sec / 3600)

    cap.release()
    cv2.destroyAllWindows()

    return {
        "exercise": "Push-ups",
        "reps": reps,
        "duration_sec": round(elapsed_sec, 2),
        "calories": round(calories, 2),
        "status": "Success" if reps >= target_reps else "Fail"
    }
