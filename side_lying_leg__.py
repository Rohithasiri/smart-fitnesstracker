import cv2
import mediapipe as mp
import numpy as np
import time

#  USER CONFIG
VIDEO_SOURCE = "/Users/rohithayenugu/Desktop/Screen Recording 2025-06-09 at 10.11.49 PM.mov"
WEIGHT_KG = 60    
MET = 3.8         
GOOD_ANGLE = 160   
# SETUP 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise IOError(f"Cannot open video source: {VIDEO_SOURCE}")

# Rep & calorie tracking
count = 0
state = 0          # 0 = lowering, 1 = lifting
threshold_up = 0.40
threshold_down = 0.55

start_time = time.time()
calories = 0.0

#  UTILITY FUNCTIONS 
def calculate_angle(a, b, c):
    """Returns the angle (deg) at joint b made by points a‑b‑c."""
    a, b, c = map(np.array, (a, b, c))
    angle = np.abs(np.degrees(
        np.arctan2(c[1]-b[1], c[0]-b[0]) -
        np.arctan2(a[1]-b[1], a[0]-b[0])
    ))
    return 360 - angle if angle > 180 else angle


def norm2pix(norm_pt, frame_shape):
    """Convert a Mediapipe (x,y) normalized point to pixel coords."""
    h, w = frame_shape[:2]
    return int(norm_pt[0] * w), int(norm_pt[1] * h)

# MAIN LOOP 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # natural mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # --- Right‑leg landmarks ---
        hip   = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee  = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                 lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        angle = calculate_angle(hip, knee, ankle)
        ankle_y = ankle[1]

        # REP LOGIC 
        if ankle_y < threshold_up and state == 0:
            state = 1  # leg is up
        elif ankle_y > threshold_down and state == 1:
            count += 1
            state = 0  # leg is down

        #  CALORIE TRACK 
        elapsed_sec = time.time() - start_time
        cals_per_min = MET * 3.5 * WEIGHT_KG / 200  # kcal/min formula
        calories = cals_per_min * (elapsed_sec / 60)

        #  FORM FEEDBACK 
        good_form = angle >= GOOD_ANGLE
        feedback_text = "correct" if good_form else "Lift Higher ↗"
        feedback_color = (0, 255, 0) if good_form else (0, 0, 255)

        # DRAWINGS 
        mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS)

        # Angle label near knee
        knee_px, knee_py = norm2pix(knee, frame.shape)
        cv2.putText(frame, f"{int(angle)}°", (knee_px + 10, knee_py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # HUD (top‑left)
        cv2.rectangle(frame, (15, 15), (260, 165), (0, 0, 0), -1)
        cv2.putText(frame, f"Reps : {count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Calories  : {calories:5.1f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, feedback_text, (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2)

    #  WINDOW DISPLAY 
    cv2.imshow("Side_Lying Leg Raise Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
