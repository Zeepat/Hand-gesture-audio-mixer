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

# Define a modern font setting
FONT = cv2.FONT_HERSHEY_PLAIN  # More modern, cleaner font
FONT_SCALE = 0.9  # Adjust scale as needed for the cleaner font

# Initialize MediaPipe Hands and Drawing utilities.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize song control variables
current_song_index = 0
songs = []
last_song_change_time = 0
song_change_cooldown = 1.0  # Cooldown in seconds to prevent rapid song changes
song_change_notification = None
song_change_notification_time = 0
song_change_display_duration = 2.0  # How long to show the song change notification

# NEW: Debug variables for gesture detection
gesture_debug_message = None
gesture_debug_time = 0
gesture_debug_duration = 2.0  # How long to show gesture detection messages
gesture_visual_debug = False  # Set to True for additional visual debugging

# NEW: Function to load and play a song at a specific index
def load_and_play_song(index):
    global current_sound, pitch_shifter, analyzer, fft_analyzer, current_song_index, song_change_notification, song_change_notification_time
    
    if not songs or index < 0 or index >= len(songs):
        logging.warning(f"Invalid song index: {index}, available songs: {len(songs)}")
        return False
    
    current_song_index = index
    song_path = songs[current_song_index]
    song_name = song_path.name
    
    logging.info(f"Loading song: {song_name} (index {current_song_index + 1}/{len(songs)})")
    
    # Set song change notification
    song_change_notification = f"Now Playing: {song_name}"
    song_change_notification_time = time.time()
    
    try:
        if using_pyo:
            # Stop current playback
            if 'pitch_shifter' in globals() and pitch_shifter is not None:
                pitch_shifter.stop()
            if 'current_sound' in globals() and current_sound is not None:
                current_sound.stop()
                
            # Load and play new song
            current_sound = SfPlayer(str(song_path), loop=True)
            pitch_shifter = Harmonizer(current_sound, transpo=0)
            analyzer = Follower(pitch_shifter)
            
            # Create or update FFT analyzer
            if 'fft_analyzer' not in globals() or fft_analyzer is None:
                fft_size = 1024
                fft_analyzer = Spectrum(pitch_shifter, size=fft_size)
            else:
                # Update FFT analyzer input
                fft_analyzer.setInput(pitch_shifter)
                
            # Start playback
            pitch_shifter.out()
            return True
        else:
            # Pygame implementation
            pygame.mixer.music.stop()
            pygame.mixer.music.load(str(song_path))
            pygame.mixer.music.play(-1)
            return True
    except Exception as e:
        logging.error(f"Error changing song: {e}")
        return False

# NEW: Functions to change to next/previous songs
def next_song():
    global current_song_index, gesture_debug_message, gesture_debug_time
    if not songs:
        logging.warning("No songs available to play")
        gesture_debug_message = "No songs available!"
        gesture_debug_time = time.time()
        return False
        
    next_index = (current_song_index + 1) % len(songs)
    logging.info(f"Changing to next song (index {next_index + 1}/{len(songs)})")
    
    gesture_debug_message = "Skip song triggered"
    gesture_debug_time = time.time()
    
    return load_and_play_song(next_index)

def previous_song():
    global current_song_index, gesture_debug_message, gesture_debug_time
    if not songs:
        logging.warning("No songs available to play")
        gesture_debug_message = "No songs available!"
        gesture_debug_time = time.time()
        return False
        
    prev_index = (current_song_index - 1) % len(songs)
    logging.info(f"Changing to previous song (index {prev_index + 1}/{len(songs)})")
    
    gesture_debug_message = "Previous song triggered"
    gesture_debug_time = time.time()
    
    return load_and_play_song(prev_index)

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
                
                # Create an analyzer for waveform visualization
                analyzer = Follower(pitch_shifter)
                
                # NEW: Create FFT analyzer for frequency spectrum analysis
                fft_size = 1024
                fft_analyzer = Spectrum(pitch_shifter, size=fft_size)
                
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

if not cap.isOpened():
    logging.error("Cannot open webcam!")
    exit(1)
logging.info("Webcam opened successfully.")

cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MediaPipe Hands', 960, 720)  # Adjust dimensions as needed

# Calibration variables
is_calibrating = True
max_distances = [0, 0]  # For left and right hand
calibration_frames = 0
total_calibration_frames = 30

# Define variables for pitch control
min_pinch_ratio = 0.5  # Pinch ratio at 0% effect
max_pinch_ratio = 2.5  # Pinch ratio at 100% effect
max_semitones = 12     # Maximum pitch shift (1 octave)
max_speed = 2.0        # Maximum playback speed (2x instead of 3x)

# For waveform animation
wave_time_offset = 0
last_update_time = time.time()

# More modern, cleaner font
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 0.9

# NEW: Function to process FFT data for visualization
def process_fft_data(fft_data, num_bins=32):
    """
    Process FFT data into a smaller number of bins for visualization
    with exaggerated amplitude response for better visibility
    
    Args:
        fft_data: The raw FFT data
        num_bins: Number of frequency bins to visualize
        
    Returns:
        List of amplitudes for each frequency bin
    """
    if fft_data is None or len(fft_data) == 0:
        return [0] * num_bins
    
    # Get magnitude (absolute value) of FFT data
    # We only need the first half of FFT data (real signal)
    magnitudes = np.abs(fft_data[:len(fft_data)//2])
    
    # If we have FFT data but no sound playing, magnitudes will be all zeros
    if np.sum(magnitudes) < 0.01:
        return [0] * num_bins
    
    # Group frequencies into logarithmic bins (more natural for audio)
    # This gives more detail to lower frequencies (which is perceptually important)
    bin_size = len(magnitudes) // num_bins
    bins = []
    
    for i in range(num_bins):
        # Use logarithmic scaling for bin ranges
        start = int((len(magnitudes) ** (i / num_bins)) - 1)
        end = int((len(magnitudes) ** ((i + 1) / num_bins)) - 1)
        
        if start >= len(magnitudes):
            start = len(magnitudes) - 1
        if end >= len(magnitudes):
            end = len(magnitudes) - 1
            
        # Ensure we have at least 1 sample per bin
        if start == end:
            bin_val = magnitudes[start]
        else:
            # Take average magnitude within the bin range
            bin_val = np.mean(magnitudes[start:end])
            
        # ENHANCED: Apply more aggressive logarithmic scaling with boosting
        # Increase factor from 10 to 20 for more dramatic effect
        bin_val = np.log10(bin_val + 1) * 20
        
        # ENHANCED: Apply frequency-dependent boost
        # Boost low frequencies (bass) more than high frequencies
        if i < num_bins // 3:  # Lower third (bass)
            freq_boost = 1.8
        elif i < 2 * num_bins // 3:  # Middle third (mids)
            freq_boost = 1.4
        else:  # Upper third (highs)
            freq_boost = 1.2
            
        bin_val *= freq_boost
        
        bins.append(bin_val)
    
    # Normalize to 0-1 range with a minimum threshold for visibility
    max_val = max(bins) if max(bins) > 0 else 1
    min_threshold = 0.15  # Ensure quiet parts are still visible
    normalized_bins = [max(min_threshold, min(val / max_val, 1.0)) for val in bins]
    
    return normalized_bins

# Initialize variables for enhanced spectrum visualization
fft_data = None
spectrum_bins = []
spectrum_smoothing = 0.6  # Reduced from 0.7 for more responsive visualization
beat_detection = 0.0  # For detecting beats in the music
last_beat_time = 0  # Time of last detected beat

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
    
    # Update wave animation time
    current_time = time.time()
    time_delta = current_time - last_update_time
    wave_time_offset += time_delta * 5  # Controls animation speed
    last_update_time = current_time
    
    # Initialize volume with a default value
    volume = 0.5  # Default to medium volume
    
    # Check if hands are detected
    midpoints = []  # Store midpoints for both hands
    hand_scales = []  # Store distances between index_mcp and pinky_mcp for each hand
    pinch_ratios = []  # Store thumb-index pinch ratios for each hand
    hand_labels = []   # Store hand labels (Left/Right)
    hand_centers = []  # NEW: Store hand palm centers for vertical position detection
    
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
            wrist = hand_landmarks.landmark[0]
            middle_mcp = hand_landmarks.landmark[9]  # Middle finger MCP joint
            
            # Convert normalized coordinates to pixel coordinates
            h, w, c = image.shape
            thumb_tip_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_tip_px = (int(index_tip.x * w), int(index_tip.y * h))
            index_mcp_px = (int(index_mcp.x * w), int(index_mcp.y * h))
            pinky_mcp_px = (int(pinky_mcp.x * w), int(pinky_mcp.y * h))
            
            # Calculate palm center (between wrist and middle finger MCP)
            palm_center_x = int((wrist.x + middle_mcp.x) * w / 2)
            palm_center_y = int((wrist.y + middle_mcp.y) * h / 2)
            palm_center = (palm_center_x, palm_center_y)
            
            # NEW: Store hand palm center for vertical position detection
            hand_centers.append((handedness, palm_center))
            
            # Draw only thumb tip and index tip landmarks (removing knuckles)
            cv2.circle(image, thumb_tip_px, 2, (255, 255, 255), -1)  # Red for thumb tip
            cv2.circle(image, index_tip_px, 2, (255, 255, 255), -1)  # Green for index tip
            
            # Draw a line between thumb tip and index tip (landmarks 4 and 8)
            cv2.line(image, thumb_tip_px, index_tip_px, (255, 255, 255), 1)
            
            # Calculate midpoint between keypoints 4 and 8
            midpoint_x = int((thumb_tip.x + index_tip.x) * w / 2)
            midpoint_y = int((thumb_tip.y + index_tip.y) * h / 2)
            midpoint = (midpoint_x, midpoint_y)
            
            # Draw the midpoint
            cv2.circle(image, midpoint, 0, (255, 165, 0), -1)  # Orange circle for midpoint
            
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
            
            # Display function of each hand instead of raw pinch ratio
            if handedness == "Left":  # Right hand in mirrored image
                speed_info = "Normal Speed"
                if pinch_ratio > min_pinch_ratio:
                    if pinch_ratio >= max_pinch_ratio:
                        speed_multiplier = max_speed
                        speed_info = f"Max Speed ({max_speed:.1f}x)"
                    else:
                        speed_effect = (pinch_ratio - min_pinch_ratio) / (max_pinch_ratio - min_pinch_ratio)
                        speed_multiplier = 1.0 + (speed_effect * (max_speed - 1.0))
                        speed_info = f"{speed_multiplier:.1f}x"
                
                # Right-align text for left hand (text ends at palm center)
                text = f"Speedup: {speed_info}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]  # Changed from 0.6, 2
                text_x = palm_center[0] - text_size[0]  # Right-aligned
                text_y = palm_center[1]
                
                cv2.putText(
                    image, 
                    text, 
                    (text_x, text_y), 
                    FONT,  # Use the new font
                    FONT_SCALE,  # Use the new scale
                    (255, 255, 255), 
                    1     # Changed from 2
                )
            
            elif handedness == "Right":  # Left hand in mirrored image
                pitch_info = "Normal Pitch"
                if pinch_ratio > min_pinch_ratio:
                    if pinch_ratio >= max_pinch_ratio:
                        semitones = max_semitones
                        pitch_info = f"+{semitones:.0f} semitones"
                    else:
                        pitch_effect = (pinch_ratio - min_pinch_ratio) / (max_pinch_ratio - min_pinch_ratio)
                        semitones = pitch_effect * max_semitones
                        pitch_info = f"+{semitones:.1f} semitones"
                
                # Left-align text for right hand (text starts at palm center)
                text = f"Pitch: {pitch_info}"
                text_x = palm_center[0]  # Left-aligned
                text_y = palm_center[1]
                
                cv2.putText(
                    image, 
                    text, 
                    (text_x, text_y), 
                    FONT,  # Use the new font
                    FONT_SCALE,  # Use the new scale
                    (255, 255, 255), 
                    1
                )
        
        # Process right hand for pitch control (even if it's the only hand)
        right_hand_index = None
        left_hand_index = None
        for i, label in enumerate(hand_labels):
            if label == "Right":  # This is left hand in mirrored image
                right_hand_index = i
            elif label == "Left":  # This is right hand in mirrored image
                left_hand_index = i
        
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
            
            # Remove on-screen text and bars for pitch control
        
        # If left hand is detected, control the playback speed
        if left_hand_index is not None:
            pinch_ratio = pinch_ratios[left_hand_index]
            
            # Map pinch ratio to speed control (0.5 to 2.5 range)
            if pinch_ratio <= min_pinch_ratio:
                speed_effect = 0.0  # Normal speed (1x)
                speed_multiplier = 1.0
                speed_effect_text = "Normal Speed"
            elif pinch_ratio >= max_pinch_ratio:
                speed_effect = 1.0  # Maximum speed (2x)
                speed_multiplier = max_speed
                speed_effect_text = "Maximum Speed"
            else:
                # Linear mapping from [0.5, 2.5] to [0, 1]
                speed_effect = (pinch_ratio - min_pinch_ratio) / (max_pinch_ratio - min_pinch_ratio)
                # Map to speed range [1.0, 2.0]
                speed_multiplier = 1.0 + (speed_effect * (max_speed - 1.0))
                
                # Set descriptive text based on effect level
                if speed_effect < 0.25:
                    speed_effect_text = "Slight Speed Up"
                elif speed_effect < 0.5:
                    speed_effect_text = "Medium Speed"
                elif speed_effect < 0.75:
                    speed_effect_text = "Fast"
                else:
                    speed_effect_text = "Very Fast"
            
            try:
                # Apply speed change based on the audio backend
                if using_pyo and current_sound is not None:
                    # For pyo, adjust the speed without changing pitch
                    current_sound.setSpeed(speed_multiplier)
                    
                    # Calculate pitch compensation to prevent "chipmunk effect"
                    # When speed increases, we need to lower the pitch proportionally
                    # 2x speed needs -12 semitones (one octave down) compensation
                    pitch_compensation = -12 * (speed_multiplier - 1)
                    
                    # Apply the combined pitch effect (user's pitch control + speed compensation)
                    if right_hand_index is not None:
                        # If right hand is active, add the compensation to the user's desired pitch
                        user_pitch = pitch_effect * max_semitones
                        pitch_shifter.setTranspo(user_pitch + pitch_compensation)
                    else:
                        # If only speed is being controlled, just apply the compensation
                        pitch_shifter.setTranspo(pitch_compensation)
                else:
                    # For pygame, we would need to reload the file with different settings
                    # This is a limitation of pygame - real-time speed change isn't easily supported
                    pass
            except Exception as e:
                logging.error(f"Error setting speed: {e}")
            
            # Remove on-screen text and bars for speed control
        
        # FIXED: Check for vertical hand position gestures for song navigation
        current_time = time.time()
        if len(hand_centers) == 2 and current_time - last_song_change_time > song_change_cooldown:
            left_hand = None
            right_hand = None
            left_hand_idx = None
            right_hand_idx = None
            
            # Identify left and right hands and track their indices
            for idx, (hand_label, center) in enumerate(hand_centers):
                if hand_label == "Left":  # This is the right hand in mirrored image
                    left_hand = center
                    left_hand_idx = idx
                elif hand_label == "Right":  # This is the left hand in mirrored image
                    right_hand = center
                    right_hand_idx = idx
            
            if left_hand and right_hand:
                # Calculate vertical distance between hands (negative if left is above right)
                vertical_distance = left_hand[1] - right_hand[1]
                
                # Define a threshold for vertical positioning (in pixels)
                # Reduced from 150 to make detection more sensitive
                vertical_threshold = 100
                
                # Display debugging info for gesture detection
                if gesture_visual_debug:
                    # Draw a line connecting the hand centers
                    cv2.line(image, left_hand, right_hand, (0, 255, 255), 2)
                    
                    # Display the vertical distance
                    mid_x = (left_hand[0] + right_hand[0]) // 2
                    mid_y = (left_hand[1] + right_hand[1]) // 2
                    
                    dist_text = f"Vert dist: {vertical_distance:.1f} px (threshold: {vertical_threshold})"
                    cv2.putText(image, dist_text, (mid_x - 100, mid_y - 20), 
                                FONT, FONT_SCALE, (0, 255, 255), 1)
                
                # Left hand over right hand = next song
                # Using negative threshold since y-coordinates increase downward
                if vertical_distance < -vertical_threshold:
                    # Visual confirmation in logs
                    logging.info(f"Gesture detected: Left hand over right hand - vertical diff: {vertical_distance:.1f}px")
                    
                    # Debug message
                    gesture_debug_message = "Left hand over right - Skip song"
                    gesture_debug_time = current_time
                    
                    # Change song
                    if next_song():
                        last_song_change_time = current_time
                
                # Right hand over left hand = previous song
                elif vertical_distance > vertical_threshold:
                    # Visual confirmation in logs
                    logging.info(f"Gesture detected: Right hand over left hand - vertical diff: {vertical_distance:.1f}px")
                    
                    # Debug message
                    gesture_debug_message = "Right hand over left - Previous song"
                    gesture_debug_time = current_time
                    
                    # Change song
                    if previous_song():
                        last_song_change_time = current_time
        
        # If we detected two hands, control volume with the distance between hands
        if len(midpoints) == 2:
            # Don't draw the simple red line - replace with waveform visualization
            # Calculate line parameters between hands
            start_point = midpoints[0]
            end_point = midpoints[1]
            
            # Calculate vector between hands
            vec_x = end_point[0] - start_point[0]
            vec_y = end_point[1] - start_point[1]
            distance = math.sqrt(vec_x**2 + vec_y**2)
            
            # Calculate perpendicular unit vector for wave amplitude direction
            if distance > 0:
                unit_x = vec_x / distance
                unit_y = vec_y / distance
                perp_x = -unit_y  # Perpendicular vector
                perp_y = unit_x
                
                # Create a wave pattern based on time and position
                num_points = 150  # Number of points in the wave
                points = []
                
                # Make amplitude depend on volume for visual feedback
                base_amplitude = 15 + volume * 45  # 15-60 pixel range based on volume
                
                # Get audio energy level if using pyo
                audio_energy = 1.0
                if using_pyo and 'analyzer' in locals():
                    try:
                        audio_energy = analyzer.get() * 5  # Scale factor
                    except:
                        pass
                
                for i in range(num_points):
                    # Position along the line
                    pos_ratio = i / (num_points - 1)
                    pos_x = int(start_point[0] + pos_ratio * vec_x)
                    pos_y = int(start_point[1] + pos_ratio * vec_y)
                    
                    # Create a dynamic wave pattern using multiple sine waves
                    # Phase shifts based on time create animation effect
                    wave1 = math.sin(wave_time_offset + i * 0.15) * base_amplitude
                    wave2 = math.sin(wave_time_offset * 2.5 + i * 0.3) * base_amplitude * 0.3
                    wave3 = math.sin(wave_time_offset * 1.7 + i * 0.05) * base_amplitude * 0.1
                    
                    # Final wave shape combines multiple frequencies
                    wave_amplitude = (wave1 + wave2 + wave3) * audio_energy
                    
                    # Apply the amplitude along the perpendicular direction
                    wave_x = int(pos_x + wave_amplitude * perp_x)
                    wave_y = int(pos_y + wave_amplitude * perp_y)
                    
                    points.append((wave_x, wave_y))
                
                # Create a copy of the image for the glow effect
                glow_layer = image.copy()
                
                # Draw the wave with a plain white color instead of gradient
                for i in range(len(points) - 1):
                    # Use white color for all line segments
                    color = (255, 255, 255)  # White in BGR
                    
                    # Draw line segment on both the original and glow layer
                    cv2.line(image, points[i], points[i+1], color, 2)
                    cv2.line(glow_layer, points[i], points[i+1], color, 8)  # Thicker for glow effect
                
                # Apply blur to create a neon glow effect
                glow_layer = cv2.GaussianBlur(glow_layer, (15, 15), 0)
                
                # Blend the glow with the original image
                cv2.addWeighted(glow_layer, 0.6, image, 0.6, 0, image)
                
                # Add sparkle effects at wave peaks for extra visual appeal
                for i in range(5, len(points) - 5, 10):
                    if abs(points[i][1] - points[i+1][1]) > base_amplitude * 0.5:
                        # Draw a small starburst/sparkle at high amplitude points
                        sparkle_size = int(5 + volume * 5)
                        cv2.circle(image, points[i], sparkle_size, (255, 255, 255), -1)
                        cv2.circle(image, points[i], sparkle_size//2, (100, 200, 255), -1)
            
            # Continue with volume control logic
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
            
            # Add volume indicator above the wave
            # Show volume as a float with one decimal place (0.0-10.0)
            vol_display = volume * 10  # Convert 0.0-1.0 to 0.0-10.0
            
            # Calculate position - midway between the two hand midpoints
            mid_x = int((midpoints[0][0] + midpoints[1][0]) / 2)
            mid_y = int((midpoints[0][1] + midpoints[1][1]) / 2) - 30  # 30 pixels above the midpoint
            
            # Create volume text with one decimal point (e.g., 2.3 or 5.7)
            vol_text = f"{vol_display:.1f}"
            
            # Get text size for centering
            text_size = cv2.getTextSize(vol_text, FONT, FONT_SCALE, 1)[0]
            text_x = mid_x - (text_size[0] // 2)  # Center text horizontally
            
            # Display just the volume number without background
            cv2.putText(image, vol_text, (text_x, mid_y), FONT, FONT_SCALE, (255, 255, 255), 1)
            
            # Remove volume bar and label
            
            # Remove pinch ratio display
    
    # NEW: Display song change notification if active
    if song_change_notification and time.time() - song_change_notification_time < song_change_display_duration:
        # Create semi-transparent overlay for the notification
        overlay = image.copy()
        
        # Calculate text position (centered at the top of the screen)
        text_size = cv2.getTextSize(song_change_notification, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = 50  # From top
        
        # Draw background rectangle
        cv2.rectangle(overlay, 
                     (text_x - 20, text_y - 40), 
                     (text_x + text_size[0] + 20, text_y + 10),
                     (0, 0, 0), -1)
        
        # Add the overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Draw the notification text
        cv2.putText(image, song_change_notification, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # NEW: Add visual instructions for song navigation at the bottom of the screen
        if time.time() - song_change_notification_time < song_change_display_duration * 2:
            instructions = "Song Navigation: Left hand over Right = Next, Right hand over Left = Previous"
            inst_size = cv2.getTextSize(instructions, FONT, FONT_SCALE, 1)[0]
            inst_x = (image.shape[1] - inst_size[0]) // 2
            inst_y = image.shape[0] - 30
            
            # Draw semi-transparent background
            cv2.rectangle(overlay, 
                         (inst_x - 10, inst_y - 20), 
                         (inst_x + inst_size[0] + 10, inst_y + 10),
                         (0, 0, 0), -1)
            
            # Add the overlay
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
            
            # Draw text
            cv2.putText(image, instructions, (inst_x, inst_y), 
                        FONT, FONT_SCALE, (255, 255, 255), 1)
    
    # NEW: Display gesture debug information
    if gesture_debug_message and time.time() - gesture_debug_time < gesture_debug_duration:
        # Create semi-transparent overlay for the debug message
        overlay = image.copy()
        
        # Position at the top center, below any potential song notification
        text_size = cv2.getTextSize(gesture_debug_message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = 100  # Below song notification
        
        # Draw a highlighted background for visibility
        cv2.rectangle(overlay, 
                     (text_x - 10, text_y - 30), 
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 100, 0), -1)  # Dark green background
        
        # Add the overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Draw the debug message with a white outline for better visibility
        # Draw outline by creating multiple offset text elements
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            cv2.putText(image, gesture_debug_message, (text_x + dx, text_y + dy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
        # Draw the main text
        cv2.putText(image, gesture_debug_message, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # NEW: Display song count and current song index when no notifications are active
    if not song_change_notification or time.time() - song_change_notification_time >= song_change_display_duration:
        if songs:
            song_info = f"Song {current_song_index + 1}/{len(songs)}: {songs[current_song_index].name}"
            
            # Make text smaller to fit on screen
            text_size = cv2.getTextSize(song_info, FONT, FONT_SCALE, 1)[0]
            max_width = image.shape[1] - 20
            
            # If text is too long, truncate it
            if text_size[0] > max_width:
                # Calculate how many characters we can fit
                chars_per_pixel = len(song_info) / text_size[0]
                max_chars = int(max_width * chars_per_pixel) - 10  # Leave some margin
                song_info = song_info[:max_chars] + "..."
            
            text_x = 10  # Left-aligned
            text_y = 25  # From top
            
            # Add semi-transparent background
            overlay = image.copy()
            text_size = cv2.getTextSize(song_info, FONT, FONT_SCALE, 1)[0]
            cv2.rectangle(overlay, 
                         (text_x - 5, text_y - 20), 
                         (text_x + text_size[0] + 5, text_y + 5),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
            
            # Draw song info
            cv2.putText(image, song_info, (text_x, text_y), 
                        FONT, FONT_SCALE, (255, 255, 255), 1)
    
    # NEW: Always show gesture control hint in bottom-right corner
    if songs:  # Only show if we have songs to control
        hint_text = "Vertical hand gestures control song navigation"
        text_size = cv2.getTextSize(hint_text, FONT, FONT_SCALE * 0.8, 1)[0]
        text_x = image.shape[1] - text_size[0] - 10  # Right-aligned
        text_y = image.shape[0] - 10  # Bottom
        
        # Add semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, 
                     (text_x - 5, text_y - 15), 
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Draw hint
        cv2.putText(image, hint_text, (text_x, text_y), 
                    FONT, FONT_SCALE * 0.8, (200, 200, 255), 1)
    
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
        if 'fft_analyzer' in locals():
            fft_analyzer.stop()
        if 's' in locals():
            s.stop()
    else:
        pygame.mixer.music.stop()
        pygame.quit()
except Exception as e:
    logging.error(f"Error stopping audio: {e}")
