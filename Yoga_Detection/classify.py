import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("yoga_pose_model.keras")

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Pose class names
class_names = ["chair", "cobra", "dog", "no_pose", "shoulder_stand", "triangle", "tree", "warrior"]

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame and extract landmarks
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        # Draw landmarks on the frame in default color
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract the landmark positions (x, y coordinates)
        landmarks = []
        for landmark in result.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y])

        # Flatten landmarks to match the model input shape
        landmarks = np.array(landmarks).flatten()  # Flatten to 1D array
        landmarks = landmarks.astype(np.float32)

        # Make a prediction using the model
        prediction = model.predict(landmarks.reshape(1, -1))  # Predict from the single frame
        predicted_class = np.argmax(prediction, axis=1)
        predicted_confidence = np.max(prediction)  # Get the confidence score for the predicted pose

        # Map the predicted class to the class name
        predicted_pose = class_names[predicted_class[0]]
        confidence_percentage = predicted_confidence * 100  # Convert to percentage

        # Display the predicted pose and its confidence
        cv2.putText(frame, f"Pose: {predicted_pose} ({confidence_percentage:.2f}%)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # If the confidence is above 80%, color landmarks green
        if confidence_percentage > 80:
            # Draw landmarks in green if confidence is above 80%
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        else:
            # Otherwise, keep the default drawing color
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the frame
    cv2.imshow("Yoga Pose Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
