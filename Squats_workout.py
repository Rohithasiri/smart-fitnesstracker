import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Squat counter variables
squat_count = 0
squat_position = None  # 'up' or 'down'
feedback_text = ""

# ✅ Use a relative or user-friendly video path
video_path = r"C:\Users\O Lakshmi reddy\Downloads\videoplayback.mp4"  # Place the video in the same folder
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose detection
        results = pose.process(image)

        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of hip, knee, and ankle (left side)
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate knee angle
            angle = calculate_angle(hip, knee, ankle)

            # Get image dimensions
            h, w, _ = image.shape
            knee_point = tuple(np.multiply(knee, [w, h]).astype(int))

            # Display the angle
            cv2.putText(image, str(int(angle)),
                        (knee_point[0] + 10, knee_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw angle arc
            cv2.ellipse(image, knee_point, (40, 40), 0, 0, int(angle), (0, 255, 0), 2)

            # Squat detection logic
            if angle < 70:
                if squat_position != 'down':
                    squat_position = 'down'
                    feedback_text = "Go deeper!" if angle > 60 else "Good squat!"
            elif angle > 160:
                if squat_position == 'down':
                    squat_position = 'up'
                    squat_count += 1
                    feedback_text = "Nice! Stand complete."
            else:
                feedback_text = "Stand up straight!"

        except Exception as e:
            feedback_text = f"Pose not detected"

        # Squat counter box
        cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
        cv2.putText(image, 'SQUATS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, str(squat_count), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        # Feedback display
        cv2.putText(image, feedback_text, (280, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the video
        cv2.imshow('Squat Counter - With Arc', image)

        # Exit with 'q' or 'Esc'
        if cv2.waitKey(30) & 0xFF in [ord('q'), 27]:
            break

cap.release()
cv2.destroyAllWindows()
