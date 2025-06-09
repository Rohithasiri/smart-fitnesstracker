import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Rep tracking variables
count = 0
direction = 0  # 0 = down, 1 = up
threshold_up = 0.40
threshold_down = 0.55

# Function to calculate angle at joint b (in degrees)
def calculate_angle(a, b, c):
    """
    a: First point  (e.g., hip)
    b: Joint where angle is calculated (e.g., knee)
    c: Third point  (e.g., ankle)
    Returns the angle at point b formed by points a–b–c.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180 / np.pi)
    return 360 - angle if angle > 180 else angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image for natural webcam view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Extract right leg landmarks
        # a = hip, b = knee, c = ankle
        hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate the angle at the knee
        angle = calculate_angle(hip, knee, ankle)
        ankle_y = ankle[1]

        # Count reps: leg lifts up and comes down
        if ankle_y < threshold_up and direction == 0:
            direction = 1
        elif ankle_y > threshold_down and direction == 1:
            count += 1
            direction = 0

        # Draw results
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(frame, f'Angle: {int(angle)}°', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, f'Reps: {count}', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Side-Lying Leg Raise Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
