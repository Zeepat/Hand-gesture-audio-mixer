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
    logging.debug("Frame converted from BGR to RGB.")

    # Mark the image as not writeable to improve performance.
    image.flags.writeable = False
    results = hands.process(image)
    logging.debug("Processed frame through MediaPipe Hands.")

    # Prepare image for drawing.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw hand landmarks if any are detected.
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw the landmarks
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get thumb tip and index finger tip coordinates
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            
            # Calculate distance between thumb and index finger
            thumb_x, thumb_y = int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0])
            index_x, index_y = int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])
            
            # Draw a line between thumb and index finger
            cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
            
            # Calculate Euclidean distance
            distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            
            # Determine which hand (assuming max 2 hands)
            hand_idx = min(idx, 1)
            
            if is_calibrating:
                # During calibration, capture the maximum distance (L shape)
                max_distances[hand_idx] = max(max_distances[hand_idx], distance)
                
                # Display calibration instructions
                cv2.putText(image, f"CALIBRATING: Make L shape with fingers ({calibration_frames}/{total_calibration_frames})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # After calibration, calculate and display percentage
                if max_distances[hand_idx] > 0:
                    # Calculate percentage (0% when touching, 100% at max L-shape distance)
                    percentage = min(100, max(0, int((distance / max_distances[hand_idx]) * 100)))
                    
                    # Display the percentage
                    cv2.putText(image, f"Hand {hand_idx+1}: {percentage}%", 
                                (thumb_x, thumb_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    logging.debug(f"Hand {hand_idx+1} - Distance: {distance:.2f}, Percentage: {percentage}%")
        
        logging.debug("Hand landmarks drawn on frame.")
    else:
        logging.debug("No hand landmarks detected.")
        
    # Handle calibration timing
    if is_calibrating:
        calibration_frames += 1
        if calibration_frames >= total_calibration_frames:
            is_calibrating = False
            logging.info(f"Calibration complete. Max distances: {max_distances}")

    # Display the output.
    cv2.imshow('MediaPipe Hands', image)
    logging.debug("Frame displayed.")

    # Exit when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info("Exit key pressed. Quitting.")
        break
    
    # Reset calibration when 'c' is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        is_calibrating = True
        max_distances = [0, 0]
        calibration_frames = 0
        logging.info("Recalibrating...")

logging.info("Releasing webcam and closing windows.")
cap.release()
cv2.destroyAllWindows()
