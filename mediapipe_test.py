import cv2
import mediapipe as mp
import logging
import time
import numpy as np
import math
import pygame
import os
from pathlib import Path
from pyo import *  # Import pyo for real-time audio processing

# Set up logging configuration.
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Hands and Drawing utilities.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize pyo audio server for real-time audio processing
try:
    logging.info("Starting pyo audio server...")
    s = Server().boot()
    s.start()
    
    # Variables for audio control
    current_sound = None
    pitch_shifter = None
    using_pyo = True
    
    # Find songs in the "Songs" folder
    songs_folder = Path("Songs")
    if not songs_folder.exists():
        logging.warning(f"Songs folder not found at {songs_folder.absolute()}")
        songs = []
    else:
        # Get all audio files
        songs = list(songs_folder.glob("*.mp3")) + list(songs_folder.glob("*.wav")) + list(songs_folder.glob("*.ogg"))
        
        if songs:
            logging.info(f"Found {len(songs)} songs. Playing: {songs[0].name}")
            try:
                # Load the first song with pyo
                current_sound = SfPlayer(str(songs[0]), loop=True)
                
                # Create a pitch shifter effect
                pitch_shifter = Harmonizer(current_sound, transpo=0)  # Start with normal pitch
                
                # Connect to output and start playback
                pitch_shifter.out()
                logging.info("Audio playback started with real-time pitch control.")
            except Exception as e:
                logging.error(f"Error loading sound with pyo: {e}")
                using_pyo = False
        else:
            logging.warning("No songs found in the Songs folder")
            using_pyo = False
except Exception as e:
    logging.error(f"Failed to initialize pyo audio server: {e}")
    using_pyo = False

# Fall back to pygame if pyo fails
if not using_pyo:
    logging.info("Using pygame for basic audio (no pitch control).")
    pygame.init()
    pygame.mixer.init()
    
    # Find songs and play with pygame
    songs_folder = Path("Songs")
    songs = []
    if songs_folder.exists():
        songs = list(songs_folder.glob("*.mp3")) + list(songs_folder.glob("*.wav")) + list(songs_folder.glob("*.ogg"))
        if songs:
            pygame.mixer.music.load(str(songs[0]))
            pygame.mixer.music.set_volume(0.5)
            pygame.mixer.music.play(-1)

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

# Define variables for pitch control
min_pinch_ratio = 0.5  # Pinch ratio at 0% effect
max_pinch_ratio = 2.5  # Pinch ratio at 100% effect
max_semitones = 12     # Maximum pitch shift (1 octave)

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
    hand_scales = []  # Store distances between index_mcp and pinky_mcp for each hand
    pinch_ratios = []  # Store thumb-index pinch ratios for each hand
    hand_labels = []   # Store hand labels (Left/Right)
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get hand classification (left or right)
            handedness = results.multi_handedness[hand_idx].classification[0].label
            hand_labels.append(handedness)
            
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
            
            # Calculate distance between landmarks 5 and 17 (index MCP and pinky MCP)
            distance_mcp = math.sqrt(
                (index_mcp.x - pinky_mcp.x)**2 + 
                (index_mcp.y - pinky_mcp.y)**2 + 
                (index_mcp.z - pinky_mcp.z)**2
            )
            
            # Calculate distance between thumb tip and index tip (pinch gesture)
            pinch_distance = math.sqrt(
                (thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2 + 
                (thumb_tip.z - index_tip.z)**2
            )
            
            # Calculate pinch ratio (normalized by hand size)
            pinch_ratio = pinch_distance / distance_mcp if distance_mcp > 0 else 0
            
            # Store the midpoint, hand scale, and pinch ratio for later use
            midpoints.append(midpoint)
            hand_scales.append(distance_mcp)
            pinch_ratios.append(pinch_ratio)
            
            # Display pinch ratio for each hand
            cv2.putText(
                image, 
                f"{handedness} Hand: {pinch_ratio:.2f}", 
                (thumb_tip_px[0] - 50, thumb_tip_px[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 255), 
                2
            )
        
        # Process right hand for pitch control (even if it's the only hand)
        right_hand_index = None
        for i, label in enumerate(hand_labels):
            if label == "Right":  # This is left hand in mirrored image
                right_hand_index = i
                break
        
        # If right hand is detected and we're using pyo, control the pitch
        if right_hand_index is not None and using_pyo and pitch_shifter is not None:
            pinch_ratio = pinch_ratios[right_hand_index]
            
            # Map pinch ratio to pitch control (0.5 to 2.5 range)
            if pinch_ratio <= min_pinch_ratio:
                pitch_effect = 0.0  # Normal pitch (0%)
                pitch_effect_text = "Normal Pitch"
            elif pinch_ratio >= max_pinch_ratio:
                pitch_effect = 1.0  # Maximum pitch (100%)
                pitch_effect_text = "Maximum Pitch"
            else:
                # Linear mapping from [0.5, 2.5] to [0, 1]
                pitch_effect = (pinch_ratio - min_pinch_ratio) / (max_pinch_ratio - min_pinch_ratio)
                
                # Set descriptive text based on effect level
                if pitch_effect < 0.25:
                    pitch_effect_text = "Slight Pitch Up"
                elif pitch_effect < 0.5:
                    pitch_effect_text = "Medium Pitch"
                elif pitch_effect < 0.75:
                    pitch_effect_text = "High Pitch"
                else:
                    pitch_effect_text = "Very High Pitch"
            
            # Calculate semitones (0 to max_semitones)
            semitones = pitch_effect * max_semitones
            
            try:
                # Apply pitch shifting in real-time
                pitch_shifter.setTranspo(semitones)
            except Exception as e:
                logging.error(f"Error setting pitch: {e}")
            
            # Display pitch effect
            cv2.putText(
                image, 
                f"Pitch: {pitch_effect_text} ({int(pitch_effect * 100)}%)", 
                (30, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 165, 255), 
                2
            )
            
            # Show semitone shift
            cv2.putText(
                image, 
                f"Semitones: +{semitones:.1f}", 
                (30, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 165, 255), 
                2
            )
            
            # Draw a pitch effect bar
            bar_x, bar_y = 30, 210
            bar_width = 300
            bar_height = 20
            # Background bar
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 255), 2)
            # Filled part based on pitch effect
            filled_width = int(pitch_effect * bar_width)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 165, 255), -1)
        
        # If we detected two hands, control volume with the distance between hands
        if len(midpoints) == 2:
            # Draw the line between midpoints
            cv2.line(image, midpoints[0], midpoints[1], (0, 0, 255), 3)  # Red line connecting midpoints
            
            # Calculate the length of the line between midpoints
            line_length = math.sqrt(
                (midpoints[0][0] - midpoints[1][0])**2 + 
                (midpoints[0][1] - midpoints[1][1])**2
            )
            
            # Calculate average hand scale (average distance between landmarks 5 and 17)
            avg_hand_scale = (hand_scales[0] + hand_scales[1]) / 2
            
            # Calculate the ratio
            ratio = line_length / (avg_hand_scale * w)  # Multiply by width to normalize
            
            # Map ratio to volume (0.0 to 1.0)
            # Anything below 1.0 is minimum volume (0.0)
            # Anything above 5.0 is maximum volume (1.0)
            # Between 1.0 and 5.0, map linearly
            if ratio < 1.0:
                volume = 0.0
            elif ratio > 5.0:
                volume = 1.0
            else:
                volume = (ratio - 1.0) / 4.0  # Map [1.0, 5.0] to [0.0, 1.0]
            
            # Apply the volume to the appropriate audio system
            if using_pyo and pitch_shifter is not None:
                # For pyo, set the amplitude multiplier
                pitch_shifter.setMul(volume)
            else:
                pygame.mixer.music.set_volume(volume)
            
            # Display the ratio and volume on the screen
            cv2.putText(
                image, 
                f"Ratio: {ratio:.2f} | Volume: {int(volume * 100)}%", 
                (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
            
            # Draw a volume bar
            bar_x, bar_y = 30, 70
            bar_width = 300
            bar_height = 20
            # Draw background bar (empty volume)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 255), 2)
            # Draw filled volume
            filled_width = int(volume * bar_width)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)
            
            # Display individual pinch ratios in a combined format
            cv2.putText(
                image, 
                f"L Pinch: {pinch_ratios[0]:.2f} | R Pinch: {pinch_ratios[1]:.2f}", 
                (30, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 165, 0), 
                2
            )
    
    # Display the image
    cv2.imshow('MediaPipe Hands', image)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean shutdown procedure
logging.info("Releasing webcam and closing windows.")
cap.release()
cv2.destroyAllWindows()

# Stop audio playback
try:
    if using_pyo:
        if pitch_shifter is not None:
            pitch_shifter.stop()
        if current_sound is not None:
            current_sound.stop()
        if 's' in locals():
            s.stop()
    else:
        pygame.mixer.music.stop()
        pygame.quit()
except Exception as e:
    logging.error(f"Error stopping audio: {e}")
