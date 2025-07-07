import cv2
import time
import mediapipe as mp
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow/MediaPipe logs
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Critical body joints to validate full-body detection
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

def run_plank(user_weight, target_seconds=30, video_path=0):
    MET = 3.0
    visibility_threshold = 0.5
    sets = 0
    in_plank = False
    start_time = None
    plank_start_time = None
    elapsed = 0.0

    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Check if full-body is visible
                joints_visible = all(
                    landmarks[j.value].visibility > visibility_threshold
                    for j in CRITICAL_JOINTS
                )

                if joints_visible:
                    posture_ok = check_plank_posture(landmarks)

                    if posture_ok:
                        if not in_plank:
                            plank_start_time = time.time()
                            in_plank = True
                        elapsed = time.time() - plank_start_time
                        if elapsed>=target_seconds:
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

                # Draw annotations
                status_text = "HOLD" if in_plank else "NONE"
                cv2.putText(frame, f"Status: {status_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if in_plank else (0, 0, 255), 2)
                cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Sets: {sets}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Plank Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Compute calories burned
    duration_hr = elapsed / 3600
    calories_burned = MET * user_weight * duration_hr
    success = elapsed >= target_seconds
    status = "Success" if success else "Fail"

    return {
        "exercise": "Plank",
        "duration_sec": round(elapsed, 2),
        "sets": sets,
        "calories": round(calories_burned, 2),
        "status": status
    }
