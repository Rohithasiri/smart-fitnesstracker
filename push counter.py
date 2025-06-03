import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
elbow_angle = 999
reps = 0
up_position = False
down_position = False

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    angle = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]) -
                       np.arctan2(a[1] - b[1], a[0] - b[0]))
    return abs(angle)

# Function to process video and detect keypoints
def detect_pose(video_path=None):
    global elbow_angle, reps, up_position, down_position

    # Use a recorded video if `video_path` is provided, otherwise use the webcam
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # Use webcam (0 is the default camera)

    if not cap.isOpened():
        print("Error: Cannot access the video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Convert back to BGR for OpenCV rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract key points
            landmarks = results.pose_landmarks.landmark
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

            # Calculate elbow angle
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # Check push-up position
            if elbow_angle > 170:  # Adjusted "up" threshold
                if down_position:
                    reps += 1
                    print(f"Push-ups completed: {reps}")
                up_position = True
                down_position = False
            elif 70 < elbow_angle < 100:  # Adjusted "down" threshold
                down_position = True
                up_position = False

        # Display push-up count on the video feed
        cv2.putText(image, f"Push-ups: {reps}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Pose Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# To use a recorded video, pass the file path as an argument
# Example: detect_pose("path_to_video.mp4")
detect_pose("video.mp4")  # Replace with your video file path