import cv2
import mediapipe as mp
import logging
import time
import numpy as np
import math

# Set up logging configuration.
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Hands and Drawing utilities.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

logging.info("Initializing MediaPipe Hands model.")
hands = mp_hands.Hands(
    static_image_mode=False,  # Use video stream mode.
    max_num_hands=2,          # Maximum number of hands to detect.
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open a connection to the webcam at index 0.
logging.info("Opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

time.sleep(2)  # Give the webcam time to initialize.

if not cap.isOpened():
    logging.error("Cannot open webcam!")
    exit(1)
logging.info("Webcam opened successfully.")

# Calibration variables
is_calibrating = True
max_distances = [0, 0]  # For left and right hand
calibration_frames = 0
total_calibration_frames = 30

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        logging.error("Frame capture failed.")
        break
    logging.debug("Frame captured.")

    # Flip the frame horizontally for a mirror effect.
    frame = cv2.flip(frame, 1)
    logging.debug("Frame flipped horizontally.")

    # Convert the BGR frame to RGB for MediaPipe.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hand landmarks
    results = hands.process(image)
    
    # Convert the image back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            # Get coordinates for thumb tip (landmark 4) and index finger tip (landmark 8)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            
            # Calculate the 3D Euclidean distance between thumb tip and index finger tip
            distance_thumb_index = math.sqrt(
                (thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2 + 
                (thumb_tip.z - index_tip.z)**2
            )
            
            # Get coordinates for index MCP (landmark 5) and pinky MCP (landmark 17)
            index_mcp = hand_landmarks.landmark[5]
            pinky_mcp = hand_landmarks.landmark[17]
            
            # Calculate the 3D Euclidean distance between index MCP and pinky MCP
            distance_mcp = math.sqrt(
                (index_mcp.x - pinky_mcp.x)**2 + 
                (index_mcp.y - pinky_mcp.y)**2 + 
                (index_mcp.z - pinky_mcp.z)**2
            )
            
            # Calculate the ratio (first distance divided by second distance)
            # Add a small epsilon to avoid division by zero
            epsilon = 1e-6
            distance_ratio = distance_thumb_index / (distance_mcp + epsilon)
            
            # Scale ratio for display (optional)
            scaled_ratio = distance_ratio * 100  # Scale by 100 for better readability
            
            # Display both distances and the ratio on the frame
            cv2.putText(
                image, 
                f"Thumb-Index: {distance_thumb_index:.2f}", 
                (50, 50 + hand_idx * 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 0), 
                2
            )
            cv2.putText(
                image, 
                f"MCP Distance: {distance_mcp:.2f}", 
                (50, 80 + hand_idx * 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 0), 
                2
            )
            cv2.putText(
                image, 
                f"Ratio: {scaled_ratio:.2f}", 
                (50, 110 + hand_idx * 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 0, 0), 
                2
            )
    
    # Display the image with the landmarks and distance information
    cv2.imshow('MediaPipe Hands', image)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

logging.info("Releasing webcam and closing windows.")
cap.release()
cv2.destroyAllWindows()
