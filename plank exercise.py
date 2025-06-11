import cv2, time
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize video capture (0 for webcam or path to video file)
cap = cv2.VideoCapture("video.mp4")  # Change to "exercise.mp4" for a video file

# MediaPipe Pose with chosen confidence thresholds
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Visibility threshold for considering a joint "seen"
visibility_threshold = 0.5

# Define critical joints (shoulders, hips, ankles) to check for full-body
CRITICAL_JOINTS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

# Variables for timing and sets
start_time = None
sets = 0
in_plank = False

def check_plank_posture(landmarks):
    # Example: check if the body is straight (simple angle check)
    import numpy as np
    def get_point(lm): return np.array([lm.x, lm.y])
    left_shoulder = get_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    left_hip = get_point(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    left_ankle = get_point(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    # Calculate angle at the hip
    v1 = left_shoulder - left_hip
    v2 = left_ankle - left_hip
    angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    # Plank if angle is roughly straight (160-180 degrees)
    return 160 < angle < 180

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Check full-body visibility: all critical joints must be above threshold
        joints_visible = all(
            landmarks[j.value].visibility > visibility_threshold
            for j in CRITICAL_JOINTS
        )

        if joints_visible:
            # Evaluate plank posture only if all critical joints are visible
            # (Assume check_plank_posture returns True if angles are correct)
            posture_ok = check_plank_posture(landmarks)  # existing plank check logic

            if posture_ok:
                if not in_plank:
                    # Starting a new plank
                    start_time = time.time()
                    in_plank = True
                # If already in plank, continue timing (no action needed here)
            else:
                # Posture lost (still full-body visible but not meeting plank form)
                if in_plank:
                    sets += 1
                    in_plank = False
                    start_time = None
        else:
            # Full-body not visible: treat as break if a plank was in progress
            if in_plank:
                sets += 1
                in_plank = False
                start_time = None
            # Do not evaluate posture in this frame

        # Calculate elapsed time if currently in plank
        elapsed = time.time() - start_time if in_plank and start_time else 0.0

        # Overlay status, time, and sets on the frame
        status_text = "HOLD" if in_plank else "NONE"
        cv2.putText(frame, f"Status: {status_text}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if in_plank else (0,0,255), 2)
        cv2.putText(frame, f"Time: {elapsed:.1f}s", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Sets: {sets}", (10,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Display the frame
    cv2.imshow("Plank Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
