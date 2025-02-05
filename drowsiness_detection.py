from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import os
import time

# Initialize mixer for sound alert
mixer.init()
mixer.music.load("music.wav")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Constants for drowsiness detection
THRESHOLD = 0.25  
FRAME_CHECK = 20 
FACE_DETECTION_INTERVAL = 5  

# Load face detection and landmark predictor
detector = dlib.get_frontal_face_detector()
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"


predictor = dlib.shape_predictor(MODEL_PATH)

# Get eye landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Open video stream with backend handling for different OS
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  


# Check if the camera is opened properly
if not cap.isOpened():
    print("Error: Camera is not accessible. Please check your webcam.")
    exit()

flag = 0
frame_count = 0
alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    frame = imutils.resize(frame, width=600)  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face every `FACE_DETECTION_INTERVAL` frames (performance improvement)
    if frame_count % FACE_DETECTION_INTERVAL == 0:
        subjects = detector(gray, 0)

    status = "ACTIVE"  
    status_color = (0, 255, 0)  

    for subject in subjects:
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract eye regions
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Calculate average EAR
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Check if EAR is below threshold
        if ear < THRESHOLD:
            flag += 1
            if flag >= FRAME_CHECK:
                status = "DROWSINESS ALERT!"
                status_color = (0, 0, 255)  
               
                if not alarm_on:
                    mixer.music.play()
                    alarm_on = True
        else:
            flag = 0
            if alarm_on:
                mixer.music.stop()  
                alarm_on = False  

    # Display status on screen
    cv2.putText(frame, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Show the output frame
    cv2.imshow("Drowsiness Detection", frame)
    
    frame_count += 1 

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources properly
cap.release()
cv2.destroyAllWindows()
