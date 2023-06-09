import cv2
import dlib
import numpy as np
import csv
import os
from datetime import datetime

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "C:\Program Files\shape_predictor_68_face_landmarks.dat"  # Update this with the correct file path
predictor = dlib.shape_predictor(predictor_path)

# Load the known face encodings and names
# ... (same as before)

# Rest of the code remains the same


# Load the known face encodings and names
jobs_image = cv2.imread("photos/jobs.png")
jobs_gray = cv2.cvtColor(jobs_image, cv2.COLOR_BGR2GRAY)
jobs_face_landmarks = predictor(jobs_gray, dlib.rectangle(0, 0, jobs_gray.shape[1], jobs_gray.shape[0]))
jobs_face_encoding = np.array([jobs_face_landmarks.part(i).x for i in range(68)] +
                              [jobs_face_landmarks.part(i).y for i in range(68)])

ratan_tata_image = cv2.imread("photos/tata.jpg")
ratan_tata_gray = cv2.cvtColor(ratan_tata_image, cv2.COLOR_BGR2GRAY)
ratan_tata_face_landmarks = predictor(ratan_tata_gray, dlib.rectangle(0, 0, ratan_tata_gray.shape[1], ratan_tata_gray.shape[0]))
ratan_tata_face_encoding = np.array([ratan_tata_face_landmarks.part(i).x for i in range(68)] +
                                    [ratan_tata_face_landmarks.part(i).y for i in range(68)])

sadmona_image = cv2.imread("photos/sadmona.jpg")
sadmona_gray = cv2.cvtColor(sadmona_image, cv2.COLOR_BGR2GRAY)
sadmona_face_landmarks = predictor(sadmona_gray, dlib.rectangle(0, 0, sadmona_gray.shape[1], sadmona_gray.shape[0]))
sadmona_face_encoding = np.array([sadmona_face_landmarks.part(i).x for i in range(68)] +
                                 [sadmona_face_landmarks.part(i).y for i in range(68)])

tesla_image = cv2.imread("photos/tesla.jpg")
tesla_gray = cv2.cvtColor(tesla_image, cv2.COLOR_BGR2GRAY)
tesla_face_landmarks = predictor(tesla_gray, dlib.rectangle(0, 0, tesla_gray.shape[1], tesla_gray.shape[0]))
tesla_face_encoding = np.array([tesla_face_landmarks.part(i).x for i in range(68)] +
                               [tesla_face_landmarks.part(i).y for i in range(68)])

sammy2_image = cv2.imread("photos/sammy2.png")
sammy2_gray = cv2.cvtColor(sammy2_image, cv2.COLOR_BGR2GRAY)
sammy2_face_landmarks = predictor(sammy2_gray, dlib.rectangle(0, 0, sammy2_gray.shape[1], sammy2_gray.shape[0]))
sammy2_face_encoding = np.array([sammy2_face_landmarks.part(i).x for i in range(68)] +
                               [sammy2_face_landmarks.part(i).y for i in range(68)])

known_face_encodings = [
    jobs_face_encoding,
    ratan_tata_face_encoding,
    sadmona_face_encoding,
    tesla_face_encoding,
    sammy2_face_encoding
]

known_face_names = [
    "jobs",
    "tata",
    "sadmona",
    "tesla",
    "sammy2"
]

students = known_face_names.copy()

# Get the current date
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open the CSV file for writing attendance
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    # Capture video frame by frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop through each face in this frame of video
    for face in faces:
        # Find facial landmarks for each detected face
        landmarks = predictor(gray, face)
        face_encoding = np.array([landmarks.part(i).x for i in range(68)] +
                                 [landmarks.part(i).y for i in range(68)])

        # Calculate Euclidean distances between the face encoding and known face encodings
        face_distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)

        # Find the index with the smallest distance
        best_match_index = np.argmin(face_distances)

        # Threshold for face recognition
        if face_distances[best_match_index] < 0:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"

        # Add the recognized name to the attendance list
        if name in students:
            students.remove(name)
            current_time = now.strftime("%H:%M:%S")
            lnwriter.writerow([name, current_time])

        # Draw a box around the face
        top, right, bottom, left = face.top(), face.right(), face.bottom(), face.left()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow("Attendance System", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
f.close()
