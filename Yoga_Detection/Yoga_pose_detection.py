import cv2
import mediapipe as mp
import numpy as np

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

# Classify the pose using joint angles
def classifyPose(landmarks, image):
    label = 'Unknown Pose'
    color = (0, 0, 255)  # Red

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

    # -----------------------------------------------
    # Check for Warrior II Pose (Relaxed)
    # -----------------------------------------------
    if (150 < left_elbow_angle < 210 and 150 < right_elbow_angle < 210 and
        70 < left_shoulder_angle < 120 and 70 < right_shoulder_angle < 120):

        # Left leg bent, right leg straight
        if (80 < left_knee_angle < 130 and 150 < right_knee_angle < 210):
            label = 'Warrior II Pose'

        # Right leg bent, left leg straight
        elif (80 < right_knee_angle < 130 and 150 < left_knee_angle < 210):
            label = 'Warrior II Pose'

    # -----------------------------------------------
    # Check for T Pose
    # -----------------------------------------------
    if (165 < left_elbow_angle < 195 and 165 < right_elbow_angle < 195 and
        80 < left_shoulder_angle < 110 and 80 < right_shoulder_angle < 110 and
        160 < left_knee_angle < 195 and 160 < right_knee_angle < 195):
        label = 'T Pose'

    # -----------------------------------------------
    # Check for Tree Pose (arms raised + one leg bent + other leg straight)
    # -----------------------------------------------
    arms_raised = (left_elbow_angle > 160 and right_elbow_angle > 160)

    left_leg_straight = left_knee_angle > 160
    right_leg_straight = right_knee_angle > 160

    left_leg_bent = left_knee_angle < 140
    right_leg_bent = right_knee_angle < 140

    bent_leg_points_down = False

    if left_leg_bent:
        bent_leg_points_down = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y > landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    elif right_leg_bent:
        bent_leg_points_down = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y > landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

    if arms_raised and ((left_leg_straight and right_leg_bent) or (right_leg_straight and left_leg_bent)) and bent_leg_points_down:
        label = 'Vrikshasana/Tree Pose'

    # Change color if pose is recognized
    if label != 'Unknown Pose':
        color = (0, 255, 0)  # Green

    # Write label on the image
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    return image, label

# Start webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color for MediaPipe
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        frame, label = classifyPose(result.pose_landmarks.landmark, frame)

    cv2.imshow("Yoga Pose Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
