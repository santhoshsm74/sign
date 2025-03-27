import cv2
import dlib
import numpy as np
import time
import os

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def enhance_image(image):
    """Enhance image quality for better matching."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    return cv2.resize(equalized, (64, 64))

def extract_iris(image, landmarks):
    """Extract iris regions from image."""
    left_eye = landmarks.parts()[36:42]
    right_eye = landmarks.parts()[42:48]

    lx, ly, lw, lh = cv2.boundingRect(np.array([(point.x, point.y) for point in left_eye]))
    rx, ry, rw, rh = cv2.boundingRect(np.array([(point.x, point.y) for point in right_eye]))

    if lw == 0 or lh == 0 or rw == 0 or rh == 0:
        return None, None

    left_iris = enhance_image(image[ly:ly + lh, lx:lx + lw])
    right_iris = enhance_image(image[ry:ry + rh, rx:rx + rw])

    return left_iris, right_iris

def calculate_similarity(image1, image2):
    """Calculate similarity between images using pixel difference."""
    difference = cv2.absdiff(image1, image2)
    return 1 - (np.sum(difference) / (255 * image1.size))

def match_iris(live_iris, stored_iris):
    """Match live iris with stored images and return similarity score."""
    return calculate_similarity(live_iris, stored_iris) > 0.90

# Open camera
cap = cv2.VideoCapture(0)

username = input("Enter your username for login: ")

# Check if user folder exists
user_folder = f"iris_dataset/{username}"
if not os.path.exists(user_folder):
    print(f"[ERROR] User '{username}' not found.")
    exit()

print("Starting iris matching...")

matched = False
start_time = time.time()

left_match_count = 0
right_match_count = 0
total_count = 60  # Total images stored during signup

while time.time() - start_time < 10:  # 10-second limit
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Unable to capture frame.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_iris, right_iris = extract_iris(frame, landmarks)
        if left_iris is None or right_iris is None:
            print("[WARNING] Unable to detect iris regions. Trying again...")
            continue

        # Left Eye Matching
        for i in range(total_count):
            stored_iris_left = cv2.imread(f"{user_folder}/{username}_left_iris_{i}.jpg", cv2.IMREAD_GRAYSCALE)
            if stored_iris_left is not None and match_iris(left_iris, stored_iris_left):
                left_match_count += 1

        # Right Eye Matching
        for i in range(total_count):
            stored_iris_right = cv2.imread(f"{user_folder}/{username}_right_iris_{i}.jpg", cv2.IMREAD_GRAYSCALE)
            if stored_iris_right is not None and match_iris(right_iris, stored_iris_right):
                right_match_count += 1

        print(f"Left Matching Count: {left_match_count} / 60 | Right Matching Count: {right_match_count} / 60")

        if left_match_count >= 35 and right_match_count >= 35:
            print("[SUCCESS] Login Successful!")
            matched = True
            break

    cv2.imshow("Login - Iris Matching", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not matched:
    print("[FAILED] Login failed! Time limit exceeded.")

cap.release()
cv2.destroyAllWindows()
