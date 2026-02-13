"""
proportion.py — Proportion Accuracy Analysis

Evaluates how accurately the sketch reproduces the geometric proportions
of the reference portrait's facial features.

Method:
    1. Using MediaPipe landmarks, calculate key facial ratios
    2. Compare ratios between reference and sketch
    3. Compute absolute deviations
    4. Convert to a proportion score (0–100)
"""

import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh Landmark Indices for Key Points
# ---------------------------------------------------------------------------
# These indices correspond to specific facial points in the 468-landmark model.

LANDMARKS = {
    # Eyes
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    # Nose
    "nose_tip": 1,
    "nose_bridge": 6,
    "nose_bottom": 2,
    # Mouth
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip": 13,
    "lower_lip": 14,
    # Face contour
    "chin": 152,
    "forehead": 10,
    "left_cheek": 234,
    "right_cheek": 454,
    # Eyebrows
    "left_brow_outer": 46,
    "right_brow_outer": 276,
}


def _dist(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_ratios(landmarks):
    """
    Compute key facial proportion ratios from landmarks.

    Ratios computed:
        1. Eye distance / face width
        2. Nose length / face height
        3. Mouth width / face width
        4. Eye-to-chin / face height (lower face ratio)
        5. Nose width (tip to bridge) / face height
        6. Mouth height / mouth width

    Args:
        landmarks: list of (x, y) tuples (at least 468 points)

    Returns:
        Dictionary of ratio names to float values
    """
    lm = {name: landmarks[idx] for name, idx in LANDMARKS.items()}

    # Face dimensions
    face_width = _dist(lm["left_cheek"], lm["right_cheek"])
    face_height = _dist(lm["forehead"], lm["chin"])

    # Avoid division by zero
    if face_width < 1 or face_height < 1:
        return None

    # Eye distance (between inner corners)
    eye_distance = _dist(lm["left_eye_inner"], lm["right_eye_inner"])

    # Nose length (bridge to tip)
    nose_length = _dist(lm["nose_bridge"], lm["nose_tip"])

    # Mouth width
    mouth_width = _dist(lm["mouth_left"], lm["mouth_right"])

    # Mouth height (upper lip to lower lip)
    mouth_height = _dist(lm["upper_lip"], lm["lower_lip"])

    # Eye midpoint to chin
    eye_mid_x = (lm["left_eye_inner"][0] + lm["right_eye_inner"][0]) / 2
    eye_mid_y = (lm["left_eye_inner"][1] + lm["right_eye_inner"][1]) / 2
    eye_to_chin = _dist((eye_mid_x, eye_mid_y), lm["chin"])

    ratios = {
        "eye_distance_ratio": eye_distance / face_width,
        "nose_length_ratio": nose_length / face_height,
        "mouth_width_ratio": mouth_width / face_width,
        "lower_face_ratio": eye_to_chin / face_height,
        "nose_face_ratio": nose_length / face_width,
        "mouth_aspect_ratio": mouth_height / max(mouth_width, 1),
    }

    return ratios


def compare_ratios(ref_ratios, sketch_ratios):
    """
    Compare facial proportion ratios between reference and sketch.

    Args:
        ref_ratios: dictionary of ratios from reference image
        sketch_ratios: dictionary of ratios from sketch image

    Returns:
        Dictionary with:
            - score: overall proportion score (0–100)
            - details: per-ratio comparison details
    """
    if ref_ratios is None or sketch_ratios is None:
        return {
            "score": 0,
            "details": [],
            "error": "Could not compute facial ratios."
        }

    details = []
    total_accuracy = 0.0

    # Human-readable names for the ratios
    ratio_names = {
        "eye_distance_ratio": "Eye Spacing",
        "nose_length_ratio": "Nose Length",
        "mouth_width_ratio": "Mouth Width",
        "lower_face_ratio": "Lower Face Proportion",
        "nose_face_ratio": "Nose-to-Face Ratio",
        "mouth_aspect_ratio": "Mouth Shape",
    }

    for key in ref_ratios:
        ref_val = ref_ratios[key]
        sketch_val = sketch_ratios.get(key, 0)

        # Calculate deviation as percentage of reference value
        if ref_val > 0:
            deviation = abs(ref_val - sketch_val) / ref_val
        else:
            deviation = 0

        # Convert to accuracy (cap deviation at 100%)
        accuracy = max(0, 1.0 - deviation) * 100

        details.append({
            "name": ratio_names.get(key, key),
            "reference": round(ref_val, 4),
            "sketch": round(sketch_val, 4),
            "deviation_pct": round(deviation * 100, 1),
            "accuracy": round(accuracy, 1)
        })

        total_accuracy += accuracy

    # Average accuracy across all ratios
    num_ratios = len(ref_ratios)
    overall_score = total_accuracy / max(num_ratios, 1)

    return {
        "score": round(overall_score, 1),
        "details": details
    }


def analyze_proportion(ref_landmarks, sketch_landmarks):
    """
    Full proportion analysis pipeline.

    Args:
        ref_landmarks: landmarks from aligned reference image
        sketch_landmarks: landmarks from aligned sketch image

    Returns:
        Dictionary with score and per-ratio details
    """
    ref_ratios = compute_ratios(ref_landmarks)
    sketch_ratios = compute_ratios(sketch_landmarks)

    return compare_ratios(ref_ratios, sketch_ratios)
