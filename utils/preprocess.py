"""
preprocess.py — Image Preprocessing Module

Handles:
- Background removal using rembg
- Face detection and 468-landmark extraction via MediaPipe FaceLandmarker (Tasks API)
- Face alignment (rotation, scale, translation normalization)
- Orchestrates the full preprocessing pipeline for a reference + sketch pair
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
import os
import urllib.request


# ---------------------------------------------------------------------------
# 2. Face Detection & Landmark Extraction (New Tasks API)
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_landmarker.task")

def ensure_model():
    """Ensure the MediaPipe face_landmarker model exists."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face_landmarker.task to {MODEL_PATH}...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Download complete.")

def detect_landmarks(img):
    """
    Detect face landmarks using MediaPipe FaceLandmarker (Tasks API).
    Extracts 478 facial landmarks (468 face + 10 iris).

    Args:
        img: BGR image (numpy array)

    Returns:
        List of (x, y) tuples in pixel coordinates, or None if no face found
    """
    ensure_model()

    # Create FaceLandmarker
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(options)

    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Create MediaPipe Image from numpy array
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Detect landmarks
    result = detector.detect(mp_image)

    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return None

    # Extract first face's landmarks as pixel coordinates
    face_landmarks = result.face_landmarks[0]
    landmarks = []
    for lm in face_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmarks.append((x, y))

    return landmarks


# ---------------------------------------------------------------------------
# 3. Face Alignment
# ---------------------------------------------------------------------------

# MediaPipe FaceMesh landmark indices for key points
LEFT_EYE_CENTER = 468    # Left iris center (refined landmarks)
RIGHT_EYE_CENTER = 473   # Right iris center (refined landmarks)
CHIN = 152               # Bottom of chin
NOSE_TIP = 1             # Nose tip

# Fallback eye landmarks (if refined not available)
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263


def _get_eye_center(landmarks, eye_center_idx, inner_idx, outer_idx):
    """Get eye center, with fallback to averaging inner+outer corners."""
    if eye_center_idx < len(landmarks):
        return landmarks[eye_center_idx]
    # Fallback: average of inner and outer corners
    inner = landmarks[inner_idx]
    outer = landmarks[outer_idx]
    return ((inner[0] + outer[0]) // 2, (inner[1] + outer[1]) // 2)


def align_face(img, landmarks, output_size=512):
    """
    Align face using eye centers and chin point.
    Normalizes rotation, scale, and translation.

    Args:
        img: BGR image (numpy array)
        landmarks: list of (x, y) tuples from detect_landmarks()
        output_size: desired output image size (square)

    Returns:
        Aligned image (numpy array), transformed landmarks
    """
    # Get key alignment points
    left_eye = _get_eye_center(landmarks, LEFT_EYE_CENTER,
                               LEFT_EYE_INNER, LEFT_EYE_OUTER)
    right_eye = _get_eye_center(landmarks, RIGHT_EYE_CENTER,
                                RIGHT_EYE_INNER, RIGHT_EYE_OUTER)
    chin = landmarks[CHIN]

    # Calculate angle between eyes for rotation correction
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Calculate eye center (midpoint between eyes)
    eye_center = (
        (left_eye[0] + right_eye[0]) / 2.0,
        (left_eye[1] + right_eye[1]) / 2.0
    )

    # Calculate scale: distance from eye center to chin
    eye_to_chin = np.sqrt(
        (eye_center[0] - chin[0]) ** 2 +
        (eye_center[1] - chin[1]) ** 2
    )
    # We want eye-to-chin to be about 45% of output size
    desired_dist = output_size * 0.45
    scale = desired_dist / max(eye_to_chin, 1)

    # Build rotation matrix (rotate + scale around eye center)
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)

    # Adjust translation so face is centered in output
    M[0, 2] += (output_size / 2.0 - eye_center[0])
    M[1, 2] += (output_size / 2.0 - eye_center[1]) + output_size * 0.05

    # Apply affine transformation
    aligned = cv2.warpAffine(
        img, M, (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Transform landmarks to aligned coordinates
    aligned_landmarks = []
    for (x, y) in landmarks:
        new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        aligned_landmarks.append((int(new_x), int(new_y)))

    return aligned, aligned_landmarks


# ---------------------------------------------------------------------------
# 4. Full Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess_pair(ref_path, sketch_path, output_size=512):
    """
    Full preprocessing pipeline for a reference photo + sketch pair.

    Steps:
        1. Load images
        2. Detect landmarks on both
        3. Align both faces
        4. Convert to grayscale

    Args:
        ref_path: path to reference portrait photo
        sketch_path: path to user's sketch/painting
        output_size: size of aligned output images

    Returns:
        Dictionary with:
            - ref_aligned: aligned reference image (BGR)
            - sketch_aligned: aligned sketch image (BGR)
            - ref_gray: aligned reference in grayscale
            - sketch_gray: aligned sketch in grayscale
            - ref_landmarks: transformed landmarks for reference
            - sketch_landmarks: transformed landmarks for sketch
            - error: error message string, or None if successful
    """
    result = {
        "ref_aligned": None,
        "sketch_aligned": None,
        "ref_gray": None,
        "sketch_gray": None,
        "ref_landmarks": None,
        "sketch_landmarks": None,
        "error": None
    }

    # Load images
    ref_img = cv2.imread(ref_path)
    sketch_img = cv2.imread(sketch_path)

    if ref_img is None:
        result["error"] = "Could not load reference image. Please check the file."
        return result
    if sketch_img is None:
        result["error"] = "Could not load sketch image. Please check the file."
        return result

    # Detect landmarks BEFORE background removal (works better on full images)
    ref_landmarks = detect_landmarks(ref_img)
    if ref_landmarks is None:
        result["error"] = ("No face detected in the reference image. "
                           "Please upload a clear portrait photo.")
        return result

    sketch_landmarks = detect_landmarks(sketch_img)
    if sketch_landmarks is None:
        result["error"] = ("No face detected in the sketch. "
                           "Please ensure the sketch has a recognizable face.")
        return result

    # Align both faces
    ref_aligned, ref_lm = align_face(ref_img, ref_landmarks, output_size)
    sketch_aligned, sketch_lm = align_face(sketch_img, sketch_landmarks, output_size)

    # Convert to grayscale
    ref_gray = cv2.cvtColor(ref_aligned, cv2.COLOR_BGR2GRAY)
    sketch_gray = cv2.cvtColor(sketch_aligned, cv2.COLOR_BGR2GRAY)

    result["ref_aligned"] = ref_aligned
    result["sketch_aligned"] = sketch_aligned
    result["ref_gray"] = ref_gray
    result["sketch_gray"] = sketch_gray
    result["ref_landmarks"] = ref_lm
    result["sketch_landmarks"] = sketch_lm

    return result
