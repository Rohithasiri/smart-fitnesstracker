import cv2
import mediapipe as mp
import numpy as np
import time

def run_yoga_pose(user_weight, target_time, pose_name, video_path=None):
    print("function call received")
    # Initialize MediaPipe pose and drawing modules
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils

    # MET values per pose (approximate)
    MET_VALUES = {
        "T Pose": 2.5,
        "Warrior II Pose": 3.0,
        "Chair Pose": 3.5
    }
    MET = MET_VALUES.get(pose_name, 3.0)

    # Helper functions
    def calculate_angle(a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(np.degrees(radians))
        return 360 - angle if angle > 180 else angle

    def is_target_pose(pose_name, landmarks):
        if not landmarks:
            return False

        lmk = mp_pose.PoseLandmark

        # Calculate all required angles
        angles = {
            "left_hip": calculate_angle(landmarks[lmk.LEFT_SHOULDER.value], landmarks[lmk.LEFT_HIP.value], landmarks[lmk.LEFT_KNEE.value]),
            "right_hip": calculate_angle(landmarks[lmk.RIGHT_SHOULDER.value], landmarks[lmk.RIGHT_HIP.value], landmarks[lmk.RIGHT_KNEE.value]),
            "left_elbow": calculate_angle(landmarks[lmk.LEFT_SHOULDER.value], landmarks[lmk.LEFT_ELBOW.value], landmarks[lmk.LEFT_WRIST.value]),
            "right_elbow": calculate_angle(landmarks[lmk.RIGHT_SHOULDER.value], landmarks[lmk.RIGHT_ELBOW.value], landmarks[lmk.RIGHT_WRIST.value]),
            "left_shoulder": calculate_angle(landmarks[lmk.LEFT_ELBOW.value], landmarks[lmk.LEFT_SHOULDER.value], landmarks[lmk.LEFT_HIP.value]),
            "right_shoulder": calculate_angle(landmarks[lmk.RIGHT_HIP.value], landmarks[lmk.RIGHT_SHOULDER.value], landmarks[lmk.RIGHT_ELBOW.value]),
            "left_knee": calculate_angle(landmarks[lmk.LEFT_HIP.value], landmarks[lmk.LEFT_KNEE.value], landmarks[lmk.LEFT_ANKLE.value]),
            "right_knee": calculate_angle(landmarks[lmk.RIGHT_HIP.value], landmarks[lmk.RIGHT_KNEE.value], landmarks[lmk.RIGHT_ANKLE.value])
        }

        # Conditions for each pose
        if pose_name == 'T Pose':
            return (165 < angles["left_elbow"] < 195 and 165 < angles["right_elbow"] < 195 and
                    80 < angles["left_shoulder"] < 110 and 80 < angles["right_shoulder"] < 110 and
                    160 < angles["left_knee"] < 195 and 160 < angles["right_knee"] < 195)

        elif pose_name == 'Warrior II Pose':
            return ((150 < angles["left_elbow"] < 210 and 150 < angles["right_elbow"] < 210 and
                     70 < angles["left_shoulder"] < 120 and 70 < angles["right_shoulder"] < 120) and
                    ((80 < angles["left_knee"] < 130 and 150 < angles["right_knee"] < 210) or
                     (80 < angles["right_knee"] < 130 and 150 < angles["left_knee"] < 210)))

        elif pose_name == 'Chair Pose':
            knees_bent = (70 < angles["left_knee"] < 140 and 70 < angles["right_knee"] < 140)
            hips_bent = (70 < angles["left_hip"] < 130 and 70 < angles["right_hip"] < 130)
            wrists_above_shoulders = (
                landmarks[lmk.LEFT_WRIST.value].y < landmarks[lmk.LEFT_SHOULDER.value].y and
                landmarks[lmk.RIGHT_WRIST.value].y < landmarks[lmk.RIGHT_SHOULDER.value].y)
            elbows_straight = (150 < angles["left_elbow"] < 195 and 150 < angles["right_elbow"] < 195)
            return knees_bent and hips_bent and wrists_above_shoulders and elbows_straight

        return False

    cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(0)
    start_time = None
    pose_held_time = 0
    pose_active = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            if is_target_pose(pose_name, landmarks):
                if not pose_active:
                    pose_active = True
                    start_time = time.time()
                else:
                    pose_held_time = time.time() - start_time

                cv2.putText(frame, f'{pose_name} - Holding: {pose_held_time:.1f}s',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if pose_held_time >= target_time:
                    success = True
                    print(f"Pose held time is: {pose_held_time} and now exiting the loop")
                    break
            else:
                pose_active = False
                pose_held_time = 0
                start_time = None

                cv2.putText(frame, f'Please perform: {pose_name}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'No person detected',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Yoga Pose Tracker', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    duration = pose_held_time
    calories = MET * user_weight * (duration / 3600)

    return {
        "exercise": "Yoga",
        "pose": pose_name,
        "duration": round(duration, 2),
        "calories": round(calories, 2),
        "success": success,
        "status": "Success" if success else "Fail"

    }
