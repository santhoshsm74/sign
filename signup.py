import cv2
import dlib
import numpy as np
import os

# Create a directory for storing iris images
dataset_path = "iris_dataset"
os.makedirs(dataset_path, exist_ok=True)

# Load Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def enhance_image(image):
    """Enhance image quality before saving."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    return cv2.resize(equalized, (64, 64))

def extract_iris(image, landmarks):
    """Extract left and right iris from detected face landmarks."""
    left_eye = landmarks.parts()[36:42]
    right_eye = landmarks.parts()[42:48]

    lx, ly, lw, lh = cv2.boundingRect(np.array([(point.x, point.y) for point in left_eye]))
    rx, ry, rw, rh = cv2.boundingRect(np.array([(point.x, point.y) for point in right_eye]))

    if lw == 0 or lh == 0 or rw == 0 or rh == 0:
        return None, None

    left_iris = enhance_image(image[ly:ly + lh, lx:lx + lw])
    right_iris = enhance_image(image[ry:ry + rh, rx:rx + rw])

    return left_iris, right_iris

# Get username
username = input("Enter your username for signup: ")
user_folder = f"{dataset_path}/{username}"
os.makedirs(user_folder, exist_ok=True)

# Initialize camera
cap = cv2.VideoCapture(0)

count = 0
total_images = 60  # Store 60 left and 60 right iris images

print("[INFO] Capturing iris images. Look at the camera...")

while count < total_images:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not accessible!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_iris, right_iris = extract_iris(frame, landmarks)

        if left_iris is None or right_iris is None:
            print("[WARNING] Could not detect iris. Trying again...")
            continue

        # Save images in the user's folder
        cv2.imwrite(f"{user_folder}/{username}_left_iris_{count}.jpg", left_iris)
        cv2.imwrite(f"{user_folder}/{username}_right_iris_{count}.jpg", right_iris)
        print(f"[INFO] Captured {count + 1}/{total_images} iris images.")

        count += 1

    cv2.imshow("Iris Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] Iris image capturing completed successfully!")
