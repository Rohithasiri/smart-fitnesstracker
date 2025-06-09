# import cv2                        
# import mediapipe as mp            
# import numpy as np                
# import math               

# #Assigning drawing_utils from mediapipe as mp_drawing
# mp_drawing = mp.solutions.drawing_utils 

# #Assigning holistic from mediapipe as mp_holistic
# mp_holistic = mp.solutions.holistic      

# #defining a function to calculate angle
# def angle_between_lines(x1, y1, x2, y2, x3, y3):         
#     slope1 = (y2 - y1) / (x2 - x1)                       
#     slope2 = (y3 - y2) / (x3 - x2)   
#     #Calculate the angle using the slopes
#     angle = math.atan2(slope2 - slope1, 1 + slope1 * slope2)   
#     return math.degrees(angle)                               
# leglift = 0   #Initialize a variable to count the number of leg lifts       
# count1 = False       
# count2 = False        
# count3 = False       
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:  
#     cap = cv2.VideoCapture(r'/Users/rohithayenugu/Desktop/Screen Recording 2025-05-06 at 2.20.10 PM.mov')  #start capturing video from webcam  
    
    
#     while cap.isOpened():               
#         ret, frame = cap.read()          
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   #convert image to RGB
#         results = holistic.process(image)  #Make a detection using the Holistic model on the image 
#         annotated_image = image.copy()     # Make a copy of the image to draw landmarks 
#         mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#         left_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]   #get coordinates of left hip 
#         right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]   #get coordinates of right hip
#         midpoint = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)   #calculate midpoint of left and right hips
#         left_knee = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE]    
#         right_knee = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE]
#         angle1 = angle_between_lines(left_knee.x, left_knee.y, midpoint[0], midpoint[1], right_knee.x, right_knee.y)
#         print("Angles :",angle1)
#         if (angle1 > 60):
#             count1 = True
#         if (count1 == True and angle1 > 100):
#             count2 = True
#         if (count2 == True and angle1 < 60):
#             count3 = True
#         if (count1 == True and count2 == True and count3 == True):
#             leglift = leglift + 1
#             count1 = False
#             count2 = False
#             count3 = False
#         lg = leglift
#         print("Leg Lift : ",leglift)
#         cv2.circle(annotated_image, (int(midpoint[0] * annotated_image.shape[1]), int(midpoint[1] * annotated_image.shape[0])), 5, (255, 0, 0), -1)
#        #check if angle is between 68.85 to 80 and display "Correct Exercise" on screen
#         if 68.85 <= angle1 <= 80:
#             cv2.putText(annotated_image, "Correct Side Lying Leg Lift Exercise", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(annotated_image, "Incorrect Side Lying Leg Lift Exercise", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.putText(annotated_image, "Angle: " + str(round(angle1, 2)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#         cv2.putText(annotated_image, "Leg Lift: " + str(round(lg, 2)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#         cv2.imshow('MediaPipe Holistic', annotated_image)
#         #Exit if the user presses the 'q' key
#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             break
#     #release all the sources        
#     cap.release()
#     cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    """
    Calculates the angle at point 'b' formed by the points 'a', 'b', and 'c'.
    Each point is a tuple of (x, y) coordinates.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Initialize variables
leglift = 0
count1 = count2 = count3 = False

# Start capturing video
cap = cv2.VideoCapture(r'/Users/rohithayenugu/Desktop/Screen Recording 2025-05-20 at 7.01.23 PM.mov')

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Convert back to BGR for rendering
        annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if pose landmarks are detected
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Extract required landmarks
            landmarks = results.pose_landmarks.landmark
            left_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE]

            # Calculate midpoint between hips
            midpoint = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)

            # Calculate angle
            angle1 = calculate_angle((left_knee.x, left_knee.y), midpoint, (right_knee.x, right_knee.y))

            # Leg lift detection logic
            if angle1 > 60:
                count1 = True
            if count1 and angle1 > 100:
                count2 = True
            if count2 and angle1 < 60:
                count3 = True
            if count1 and count2 and count3:
                leglift += 1
                count1 = count2 = count3 = False

            # Display feedback
            if 68.85 <= angle1 <= 100:
                feedback = "Correct Side Lying Leg Lift Exercise"
                color = (0, 255, 0)
            else:
                feedback = "Incorrect Side Lying Leg Lift Exercise"
                color = (0, 0, 255)

            cv2.putText(annotated_image, feedback, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(annotated_image, f"Angle: {angle1:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(annotated_image, f"Leg Lift: {leglift}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw midpoint
            h, w, _ = annotated_image.shape
            cv2.circle(annotated_image, (int(midpoint[0] * w), int(midpoint[1] * h)), 5, (255, 0, 0), -1)

        # Display the annotated image
        cv2.imshow('MediaPipe Holistic', annotated_image)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
