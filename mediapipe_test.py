import cv2
import mediapipe as mp
import logging
import time
import numpy as np
import math
import pygame
import os
from pathlib import Path

# Set up logging configuration.
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Hands and Drawing utilities.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize pygame for audio playback
pygame.init()
pygame.mixer.init()

# Find songs in the "Songs" folder
songs_folder = Path("Songs")
if not songs_folder.exists():
    logging.warning(f"Songs folder not found at {songs_folder.absolute()}")
    songs = []
else:
    # Get all audio files (assuming common audio formats)
    songs = list(songs_folder.glob("*.mp3")) + list(songs_folder.glob("*.wav")) + list(songs_folder.glob("*.ogg"))
    
    if songs:
        logging.info(f"Found {len(songs)} songs. Playing: {songs[0].name}")
        pygame.mixer.music.load(str(songs[0]))
        # Set the volume to 50%
        pygame.mixer.music.set_volume(0.5)
        logging.info("Setting initial volume to 50%")
        pygame.mixer.music.play()
    else:
        logging.warning("No songs found in the Songs folder")

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
    
    # Convert back to BGR for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Check if hands are detected
    midpoints = []  # Store midpoints for both hands
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get the specific landmarks we want to show
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            index_mcp = hand_landmarks.landmark[5]
            pinky_mcp = hand_landmarks.landmark[17]
            
            # Convert normalized coordinates to pixel coordinates
            h, w, c = image.shape
            thumb_tip_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_tip_px = (int(index_tip.x * w), int(index_tip.y * h))
            index_mcp_px = (int(index_mcp.x * w), int(index_mcp.y * h))
            pinky_mcp_px = (int(pinky_mcp.x * w), int(pinky_mcp.y * h))
            
            # Draw only the specified landmarks
            cv2.circle(image, thumb_tip_px, 8, (0, 0, 255), -1)  # Red for thumb tip
            cv2.circle(image, index_tip_px, 8, (0, 255, 0), -1)  # Green for index tip
            cv2.circle(image, index_mcp_px, 8, (255, 0, 0), -1)  # Blue for index MCP
            cv2.circle(image, pinky_mcp_px, 8, (255, 0, 255), -1)  # Purple for pinky MCP
            
            # Draw a line between index MCP and pinky MCP (landmarks 5 and 17)
            cv2.line(image, index_mcp_px, pinky_mcp_px, (255, 255, 0), 2)
            
            # Draw a line between thumb tip and index tip (landmarks 4 and 8)
            cv2.line(image, thumb_tip_px, index_tip_px, (0, 255, 255), 2)
            
            # Calculate midpoint between keypoints 4 and 8
            midpoint_x = int((thumb_tip.x + index_tip.x) * w / 2)
            midpoint_y = int((thumb_tip.y + index_tip.y) * h / 2)
            midpoint = (midpoint_x, midpoint_y)
            
            # Draw the midpoint
            cv2.circle(image, midpoint, 10, (255, 165, 0), -1)  # Orange circle for midpoint
            
            # Store the midpoint for later use
            midpoints.append(midpoint)
        
        # If we detected two hands, connect their midpoints with a line
        if len(midpoints) == 2:
            cv2.line(image, midpoints[0], midpoints[1], (0, 0, 255), 3)  # Red line connecting midpoints
    
    # Display the image
    cv2.imshow('MediaPipe Hands', image)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

logging.info("Releasing webcam and closing windows.")
cap.release()
cv2.destroyAllWindows()

# Stop music playback
pygame.mixer.music.stop()
pygame.quit()
