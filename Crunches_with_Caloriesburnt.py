import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(r"C:\Users\O Lakshmi reddy\Downloads\crunches 1.mp4")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

counter = 0
stage = None
start_time = time.time()
calories_per_rep = 0.35
calories_burned = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    h, w, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
               lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
        knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]

        angle = calculate_angle(shoulder, hip, knee)
        print(f"Angle: {int(angle)}")  # Debug

        # Adjusted thresholds
        if angle > 120:
            stage = "down"
        if angle < 110 and stage == "down":
            stage = "up"
            counter += 1
            calories_burned = round(counter * calories_per_rep, 2)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    elapsed_time = round(time.time() - start_time, 1)
    cv2.rectangle(image, (0, 0), (500, 120), (0, 0, 0), -1)
    cv2.putText(image, f'Crunches: {counter}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(image, f'Calories: {calories_burned}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(image, f'   Time: {elapsed_time}s', (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    cv2.imshow("Crunches Tracker", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
