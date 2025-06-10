import cv2
import mediapipe as mp
import numpy as np
import time

# Constants
MET = 6.0               # MET for lunges (moderate effort)
USER_WEIGHT_KG = 60     # Change to your weight (kg)

# Initialize MediaPipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

# Initialize counters and timer
counter = 0
stage = None
start_time = None

cap = cv2.VideoCapture(r'/Users/rohithayenugu/Desktop/Screen Recording 2025-06-09 at 10.33.46 PM.mov')

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process pose
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape

        try:
            landmarks = results.pose_landmarks.landmark

            # Get right leg landmarks
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)
            knee_pos = tuple(np.multiply(knee, [w, h]).astype(int))

            # Display live angle near knee
            cv2.putText(image, f'Angle: {int(angle)}°',
                        (knee_pos[0] - 50, knee_pos[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            # Start timer on first valid detection
            if start_time is None:
                start_time = time.time()

            # Rep counting logic
            if angle < 90:
                stage = "down"
            elif angle > 160 and stage == "down":
                stage = "up"
                counter += 1

            # Form check
            if 80 < angle < 100 or angle > 160:
                form = "Correct"
                color = (0, 255, 0)
            else:
                form = "Incorrect"
                color = (0, 0, 255)

        except:
            form = "N/A"
            color = (0, 0, 255)

        # Calculate elapsed time and calories
        if start_time:
            elapsed_time_sec = time.time() - start_time
            elapsed_time_hr = elapsed_time_sec / 3600
            calories_burned = MET * USER_WEIGHT_KG * elapsed_time_hr
        else:
            elapsed_time_sec = 0
            calories_burned = 0

        # Draw UI panel
        cv2.rectangle(image, (0, 0), (350, 140), (0, 0, 0), -1)
        cv2.putText(image, f'Reps: {counter}', (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(image, f'Form: {form}', (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        cv2.putText(image, f'Time: {int(elapsed_time_sec)}s', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 200), 2)
        cv2.putText(image, f'Calories: {calories_burned:.2f}', (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the image
        cv2.imshow('Lunges Tracker - Angle, Reps, Form & Calories', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
