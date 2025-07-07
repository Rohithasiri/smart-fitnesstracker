import cv2
import mediapipe as mp
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow/MediaPipe logs
def run_lunges(user_weight, target_reps=10, video_path=0):
    """
    Runs the lunges tracker.
    
    Args:
        user_weight (float): User's weight in kg.
        target_reps (int): Number of repetitions to complete.
        video_path (int or str): 0 for webcam, or path to video file.
    
    Returns:
        dict: Summary of workout (reps, calories, success/fail)
    """

    # Constants
    MET = 6.0  # MET for lunges (moderate effort)
    
    # MediaPipe setup
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(np.degrees(radians))
        return 360 - angle if angle > 180 else angle

    # Tracking variables
    counter = 0
    stage = None
    start_time = None

    cap = cv2.VideoCapture(video_path)

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert color
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            h, w, _ = image.shape

            try:
                landmarks = results.pose_landmarks.landmark

                # Right leg points
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)
                knee_pos = tuple(np.multiply(knee, [w, h]).astype(int))

                # Display angle
                cv2.putText(image, f'Angle: {int(angle)}°',
                            (knee_pos[0] - 50, knee_pos[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                # Start timer
                if start_time is None:
                    start_time = time.time()

                # Rep counting
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

            # Elapsed time and calories
            elapsed_time_sec = time.time() - start_time if start_time else 0
            elapsed_time_hr = elapsed_time_sec / 3600
            calories_burned = MET * user_weight * elapsed_time_hr

            # UI panel
            cv2.rectangle(image, (0, 0), (350, 140), (0, 0, 0), -1)
            cv2.putText(image, f'Reps: {counter}', (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(image, f'Form: {form}', (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            cv2.putText(image, f'Time: {int(elapsed_time_sec)}s', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 200), 2)
            cv2.putText(image, f'Calories: {calories_burned:.2f}', (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)

            # Draw pose
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display window
            cv2.imshow('Lunges Tracker - Angle, Reps, Form & Calories', image)

            # Exit conditions
            if counter >= target_reps:
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return {
        "exercise": "Lunges",
        "reps": counter,
        "calories": round(calories_burned, 2),
        "status": "Success" if counter >= target_reps else "Fail"
    }


#run_lunges(62,2,'Sample Videos\Plunges.mp4')