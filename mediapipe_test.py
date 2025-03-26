import cv2
import mediapipe as mp
import logging
import time

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
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        logging.debug("Hand landmarks drawn on frame.")
    else:
        logging.debug("No hand landmarks detected.")

    # Display the output.
    cv2.imshow('MediaPipe Hands', image)
    logging.debug("Frame displayed.")

    # Exit when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info("Exit key pressed. Quitting.")
        break

logging.info("Releasing webcam and closing windows.")
cap.release()
cv2.destroyAllWindows()
