import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe pose and drawing modules
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Calculate angle between 3 points
def calculateAngle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

# Check if pose matches the target
def isTargetPose(pose_name, landmarks):
    if not landmarks:
        return False

    left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])

    right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])

    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # T Pose
    if pose_name == 'T Pose':
        if (165 < left_elbow_angle < 195 and 165 < right_elbow_angle < 195 and
            80 < left_shoulder_angle < 110 and 80 < right_shoulder_angle < 110 and
            160 < left_knee_angle < 195 and 160 < right_knee_angle < 195):
            return True

    # Warrior II Pose
    if pose_name == 'Warrior II Pose':
        if (150 < left_elbow_angle < 210 and 150 < right_elbow_angle < 210 and
            70 < left_shoulder_angle < 120 and 70 < right_shoulder_angle < 120):
            if (80 < left_knee_angle < 130 and 150 < right_knee_angle < 210):
                return True
            elif (80 < right_knee_angle < 130 and 150 < left_knee_angle < 210):
                return True
    
        # Chair Pose
    if pose_name == 'Chair Pose':
        # Check for bent knees
        knees_bent = (70 < left_knee_angle < 140 and 70 < right_knee_angle < 140)

        # Check for bent hips (leaning forward)
        hips_bent = (70 < left_hip_angle < 130 and 70 < right_hip_angle < 130)

        # Arms extended forward (elbows ~straight, wrists at/above shoulder level)
        arms_forward = (
            150 < left_elbow_angle < 195 and 150 < right_elbow_angle < 195 and
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y and
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        )

        if knees_bent and hips_bent and arms_forward:
            return True



    return False



cap = cv2.VideoCapture(3)

pose_name = 'Warrior II Pose'  # <-- change to 'Warrior II Pose' or others
start_time = None
pose_held_time = 0
pose_active = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark

        if isTargetPose(pose_name, landmarks):
            if not pose_active:
                pose_active = True
                start_time = time.time()
            else:
                pose_held_time = time.time() - start_time

            cv2.putText(frame, f'{pose_name} - Holding: {pose_held_time:.1f}s',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            if pose_active:
                pose_active = False
                pose_held_time = 0
                start_time = None

            cv2.putText(frame, f'Please perform: {pose_name}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    else:
        cv2.putText(frame, 'No person detected',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Yoga Pose Timer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
