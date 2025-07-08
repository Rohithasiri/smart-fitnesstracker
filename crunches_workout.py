import cv2
import mediapipe as mp
import numpy as np
import time

# Constants
USER_WEIGHT_KG = 60
MET = 3.5  # MET value for moderate crunches

# Initialize mediapipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open video (change to your crunches video if needed)
cap = cv2.VideoCapture(r"C:\Users\O Lakshmi reddy\Downloads\crunches 1.mp4")  # Use 0 for webcam or path to a video file

# State variables
counter = 0
stage = None
start_time = None

# Helper: calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

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

        # Use right side body landmarks
        shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
               lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
        knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]

        angle = calculate_angle(shoulder, hip, knee)

        # Draw angle near hip
        cv2.putText(image, f'Angle: {int(angle)}', (int(hip[0])-40, int(hip[1])-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Simple crunch detection logic
        if angle < 100:
            stage = "down"
        elif angle > 145 and stage == "down":
            stage = "up"
            counter += 1

        # Start time tracking
        if start_time is None:
            start_time = time.time()

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Time and calories
    elapsed = time.time() - start_time if start_time else 0
    calories = MET * USER_WEIGHT_KG * (elapsed / 3600)

    # Overlay UI
    cv2.rectangle(image, (0, 0), (360, 130), (0, 0, 0), -1)
    cv2.putText(image, f'Crunches: {counter}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(image, f'Time: {int(elapsed)}s', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 255, 200), 2)
    cv2.putText(image, f'Calories: {calories:.2f}', (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 255), 2)

    cv2.imshow("Crunches Tracker", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
