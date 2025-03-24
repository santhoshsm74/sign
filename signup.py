import cv2
import dlib
import numpy as np
import os

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, kernel)
    return sharpened

def extract_iris(image, landmarks):
    left_eye = landmarks.parts()[36:42]
    right_eye = landmarks.parts()[42:48]

    left_eye_points = [(point.x, point.y) for point in left_eye]
    right_eye_points = [(point.x, point.y) for point in right_eye]

    left_eye_np = np.array(left_eye_points, np.int32)
    right_eye_np = np.array(right_eye_points, np.int32)

    lx, ly, lw, lh = cv2.boundingRect(left_eye_np)
    rx, ry, rw, rh = cv2.boundingRect(right_eye_np)

    left_iris = enhance_image(image[ly:ly + lh, lx:lx + lw])
    right_iris = enhance_image(image[ry:ry + rh, rx:rx + rw])

    left_iris = cv2.resize(left_iris, (64, 64))
    right_iris = cv2.resize(right_iris, (64, 64))

    return left_iris, right_iris

# Initialize webcam
cap = cv2.VideoCapture(0)
count = 0

username = input("Enter your username for signup: ")

# Create a directory for the user inside iris_dataset
user_folder = f"iris_dataset/{username}"
os.makedirs(user_folder, exist_ok=True)

while count < 60:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working properly.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_iris, right_iris = extract_iris(frame, landmarks)

        cv2.imshow("Left Iris", left_iris)
        cv2.imshow("Right Iris", right_iris)

        cv2.imwrite(f"{user_folder}/{username}_left_iris_{count}.jpg", left_iris)
        cv2.imwrite(f"{user_folder}/{username}_right_iris_{count}.jpg", right_iris)
        print(f"Stored {username}_left_iris_{count}.jpg and {username}_right_iris_{count}.jpg")
        count += 1

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Iris image capturing completed.")
