import cv2
import dlib
import numpy as np
import time
import os

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def enhance_image(image):
    """Enhance the image quality."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, kernel)
    return sharpened

def extract_iris(image, landmarks):
    """Extract the iris regions from the image."""
    left_eye = landmarks.parts()[36:42]
    right_eye = landmarks.parts()[42:48]

    left_eye_points = [(point.x, point.y) for point in left_eye]
    right_eye_points = [(point.x, point.y) for point in right_eye]

    left_eye_np = np.array(left_eye_points, np.int32)
    right_eye_np = np.array(right_eye_points, np.int32)

    lx, ly, lw, lh = cv2.boundingRect(left_eye_np)
    rx, ry, rw, rh = cv2.boundingRect(right_eye_np)

    # Ensure the extracted region is not empty
    if lw == 0 or lh == 0 or rw == 0 or rh == 0:
        return None, None

    left_iris = image[ly:ly + lh, lx:lx + lw]
    right_iris = image[ry:ry + rh, rx:rx + rw]

    if left_iris.size == 0 or right_iris.size == 0:
        return None, None

    left_iris = enhance_image(left_iris)
    right_iris = enhance_image(right_iris)

    left_iris = cv2.resize(left_iris, (64, 64))
    right_iris = cv2.resize(right_iris, (64, 64))

    return left_iris, right_iris

def calculate_similarity(image1, image2):
    """Calculate the similarity between two images."""
    difference = cv2.absdiff(image1, image2)
    similarity = 1 - (np.sum(difference) / (255 * image1.size))
    return similarity

def match_iris(live_iris, stored_iris):
    """Match the live iris with stored iris images."""
    similarity = calculate_similarity(live_iris, stored_iris)
    return similarity > 0.90

# Start the camera for login
cap = cv2.VideoCapture(0)

username = input("Enter your username for login: ")

# Check if the user's folder exists
user_folder = f"iris_dataset/{username}"
if not os.path.exists(user_folder):
    print(f"[ERROR] User '{username}' not found.")
    exit()

print("Starting iris matching...")

matched = False
start_time = time.time()

left_match_count = 0
right_match_count = 0
total_count = 60  # Fixed count as per signup images

while time.time() - start_time < 10:  # Limit to 10 seconds
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Unable to capture frame.")
        continue

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract and enhance iris regions
        left_iris, right_iris = extract_iris(frame, landmarks)
        if left_iris is None or right_iris is None:
            print("[WARNING] Unable to detect iris regions. Trying again...")
            continue

        # Left Iris Matching
        for i in range(total_count):
            stored_iris_left = cv2.imread(f"{user_folder}/{username}_left_iris_{i}.jpg", cv2.IMREAD_GRAYSCALE)
            if stored_iris_left is not None and match_iris(left_iris, stored_iris_left):
                left_match_count += 1

        # Right Iris Matching
        for i in range(total_count):
            stored_iris_right = cv2.imread(f"{user_folder}/{username}_right_iris_{i}.jpg", cv2.IMREAD_GRAYSCALE)
            if stored_iris_right is not None and match_iris(right_iris, stored_iris_right):
                right_match_count += 1

        print(f"Left Matching Count: {left_match_count} / 60 | Right Matching Count: {right_match_count} / 60")

        # Check matching condition
        if left_match_count >= 25 or right_match_count >= 25:
            print("[SUCCESS] Login Successful!")
            matched = True
            break

    cv2.imshow("Login - Iris Matching", frame)

    # Break with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not matched:
    print("[FAILED] Login failed! Time limit exceeded.")

cap.release()
cv2.destroyAllWindows()
