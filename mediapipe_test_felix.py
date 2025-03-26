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

# Variables for hand tracking
left_hand_landmarks = None
right_hand_landmarks = None

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
    
    # Reset hand landmarks
    left_hand_landmarks = None
    right_hand_landmarks = None
    
    # Check if hands are detected
    if results.multi_hand_landmarks:
        # Determine left and right hands
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness classification (left or right)
            handedness = results.multi_handedness[idx].classification[0].label
            
            # Store landmarks based on handedness
            if handedness == "Left":  # This is actually right hand in the mirrored image
                right_hand_landmarks = hand_landmarks
            else:  # "Right", which is actually left hand in the mirrored image
                left_hand_landmarks = hand_landmarks
            
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            # Get coordinates for thumb tip (landmark 4) and index finger tip (landmark 8)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            
            # Convert normalized coordinates to pixel coordinates
            h, w, c = image.shape
            thumb_tip_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_tip_px = (int(index_tip.x * w), int(index_tip.y * h))
            
            # Draw blue line between thumb and index finger
            cv2.line(image, thumb_tip_px, index_tip_px, (255, 0, 0), 3)  # Blue color (BGR)
            
            # Calculate and display the distance between thumb and index
            distance_thumb_index = math.sqrt(
                (thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2
            )
            
            # Display on the frame
            cv2.putText(
                image, 
                f"{handedness} Hand: {distance_thumb_index:.2f}", 
                (thumb_tip_px[0], thumb_tip_px[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 0, 0), 
                2
            )
        
        # If both hands are detected, draw orange line between palms
        if left_hand_landmarks and right_hand_landmarks:
            # Get wrist points (bottom of palm, landmark 0)
            left_wrist = left_hand_landmarks.landmark[0]
            right_wrist = right_hand_landmarks.landmark[0]
            
            # Convert to pixel coordinates
            left_wrist_px = (int(left_wrist.x * w), int(left_wrist.y * h))
            right_wrist_px = (int(right_wrist.x * w), int(right_wrist.y * h))
            
            # Draw orange line between palm bottoms
            cv2.line(image, left_wrist_px, right_wrist_px, (0, 165, 255), 3)  # Orange color (BGR)
            
            # Calculate 3D distance between palms
            palm_distance = math.sqrt(
                (left_wrist.x - right_wrist.x)**2 + 
                (left_wrist.y - right_wrist.y)**2 +
                (left_wrist.z - right_wrist.z)**2
            )
            
            # Calculate hand size for scaling (using pinky MCP to index MCP distance on left hand)
            if left_hand_landmarks:
                index_mcp = left_hand_landmarks.landmark[5]
                pinky_mcp = left_hand_landmarks.landmark[17]
                hand_width = math.sqrt(
                    (index_mcp.x - pinky_mcp.x)**2 + 
                    (index_mcp.y - pinky_mcp.y)**2 +
                    (index_mcp.z - pinky_mcp.z)**2
                )
                
                # Calculate ratio of palm distance to hand width
                distance_ratio = palm_distance / hand_width if hand_width > 0 else 0
                
                # Display the scaled ratio
                mid_point = (
                    (left_wrist_px[0] + right_wrist_px[0]) // 2,
                    (left_wrist_px[1] + right_wrist_px[1]) // 2
                )
                cv2.putText(
                    image, 
                    f"Palm Distance Ratio: {distance_ratio:.2f}", 
                    (mid_point[0] - 100, mid_point[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 165, 255), 
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
