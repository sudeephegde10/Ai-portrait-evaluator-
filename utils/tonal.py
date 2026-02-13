"""
tonal.py — Tonal Accuracy Analysis

Evaluates how well the sketch reproduces the light and shadow patterns
of the reference portrait.

Method:
    1. Using landmarks, define facial regions (forehead, eyes, nose, cheeks, chin)
    2. Compute mean intensity per region for both images
    3. Compare region-wise brightness differences
    4. Generate tonal score (0–100)
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Facial Region Definitions (using MediaPipe landmark indices)
# ---------------------------------------------------------------------------
# Each region is defined by a set of landmark indices that form a polygon.

FACE_REGIONS = {
    "forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454,
                 323, 361, 288, 397, 365, 379, 378, 400, 377,
                 152, 148, 176, 149, 150, 136, 172, 58, 132,
                 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "left_eye": [33, 246, 161, 160, 159, 158, 157, 173,
                 133, 155, 154, 153, 145, 144, 163, 7],
    "right_eye": [362, 398, 384, 385, 386, 387, 388, 466,
                  263, 249, 390, 373, 374, 380, 381, 382],
    "nose": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2,
             164, 0, 11, 12, 13, 14, 15, 16, 17, 18,
             200, 199, 175, 152],
    "left_cheek": [234, 93, 132, 58, 172, 136, 150, 149,
                   176, 148, 152, 377, 400, 378, 379],
    "right_cheek": [454, 323, 361, 288, 397, 365, 379, 378,
                    400, 377, 152, 148, 176, 149, 150],
    "chin": [152, 377, 400, 378, 379, 365, 397, 288, 361,
             323, 454, 234, 93, 132, 58, 172, 136, 150,
             149, 176, 148],
}

# Simplified regions using bounding boxes from key landmarks
SIMPLE_REGIONS = {
    "forehead": {"top": 10, "bottom": 6, "left": 234, "right": 454},
    "left_eye_area": {"top": 159, "bottom": 145, "left": 33, "right": 133},
    "right_eye_area": {"top": 386, "bottom": 374, "left": 263, "right": 362},
    "nose": {"top": 6, "bottom": 2, "left": 219, "right": 439},
    "left_cheek": {"top": 116, "bottom": 187, "left": 234, "right": 133},
    "right_cheek": {"top": 345, "bottom": 411, "left": 362, "right": 454},
    "mouth": {"top": 13, "bottom": 14, "left": 61, "right": 291},
    "chin": {"top": 14, "bottom": 152, "left": 172, "right": 397},
}


def _get_region_bbox(landmarks, region_def):
    """
    Get a bounding box for a facial region from landmark indices.

    Args:
        landmarks: list of (x, y) tuples
        region_def: dict with 'top', 'bottom', 'left', 'right' landmark indices

    Returns:
        (x1, y1, x2, y2) bounding box, or None if invalid
    """
    try:
        top_pt = landmarks[region_def["top"]]
        bottom_pt = landmarks[region_def["bottom"]]
        left_pt = landmarks[region_def["left"]]
        right_pt = landmarks[region_def["right"]]

        x1 = max(0, left_pt[0] - 10)
        y1 = max(0, top_pt[1] - 10)
        x2 = right_pt[0] + 10
        y2 = bottom_pt[1] + 10

        # Ensure valid box
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)
    except (IndexError, KeyError):
        return None


def compute_region_intensity(gray_img, bbox):
    """
    Compute mean intensity of a region defined by a bounding box.

    Args:
        gray_img: grayscale image
        bbox: (x1, y1, x2, y2) bounding box

    Returns:
        Mean intensity (0–255), or None if region is invalid
    """
    x1, y1, x2, y2 = bbox
    h, w = gray_img.shape

    # Clamp to image bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    region = gray_img[y1:y2, x1:x2]
    if region.size == 0:
        return None

    return float(np.mean(region))


def analyze_tonal(ref_gray, sketch_gray, ref_landmarks, sketch_landmarks):
    """
    Full tonal analysis pipeline.

    Compares mean brightness in facial regions between reference and sketch.

    Args:
        ref_gray: aligned reference grayscale image
        sketch_gray: aligned sketch grayscale image
        ref_landmarks: landmarks for reference image
        sketch_landmarks: landmarks for sketch image

    Returns:
        Dictionary with:
            - score: tonal score (0–100)
            - details: per-region comparison details
    """
    details = []
    total_accuracy = 0.0
    valid_regions = 0

    # Human-readable names
    region_display_names = {
        "forehead": "Forehead",
        "left_eye_area": "Left Eye Area",
        "right_eye_area": "Right Eye Area",
        "nose": "Nose",
        "left_cheek": "Left Cheek",
        "right_cheek": "Right Cheek",
        "mouth": "Mouth",
        "chin": "Chin",
    }

    for region_name, region_def in SIMPLE_REGIONS.items():
        # Get bounding boxes using respective landmarks
        ref_bbox = _get_region_bbox(ref_landmarks, region_def)
        sketch_bbox = _get_region_bbox(sketch_landmarks, region_def)

        if ref_bbox is None or sketch_bbox is None:
            continue

        # Compute mean intensity
        ref_intensity = compute_region_intensity(ref_gray, ref_bbox)
        sketch_intensity = compute_region_intensity(sketch_gray, sketch_bbox)

        if ref_intensity is None or sketch_intensity is None:
            continue

        # Calculate brightness difference (normalized to 0–1 range)
        brightness_diff = abs(ref_intensity - sketch_intensity) / 255.0

        # Convert to accuracy score
        accuracy = max(0, (1.0 - brightness_diff * 2)) * 100  # Amplify differences

        display_name = region_display_names.get(region_name, region_name)

        details.append({
            "region": display_name,
            "ref_intensity": round(ref_intensity, 1),
            "sketch_intensity": round(sketch_intensity, 1),
            "difference": round(brightness_diff * 100, 1),
            "accuracy": round(accuracy, 1),
            "feedback": _get_tonal_feedback(display_name, ref_intensity,
                                            sketch_intensity, brightness_diff)
        })

        total_accuracy += accuracy
        valid_regions += 1

    # Compute overall tonal score
    if valid_regions > 0:
        overall_score = total_accuracy / valid_regions
    else:
        overall_score = 50.0  # Default when no regions detected

    return {
        "score": round(overall_score, 1),
        "details": details
    }


def _get_tonal_feedback(region_name, ref_val, sketch_val, diff):
    """
    Generate human-readable feedback for a tonal region comparison.

    Args:
        region_name: display name of the region
        ref_val: reference intensity
        sketch_val: sketch intensity
        diff: normalized difference (0–1)

    Returns:
        Feedback string
    """
    if diff < 0.05:
        return f"{region_name}: Excellent tonal match! ✓"
    elif diff < 0.15:
        direction = "darker" if sketch_val < ref_val else "lighter"
        return f"{region_name}: Slightly {direction} than reference. Minor adjustment needed."
    elif diff < 0.30:
        direction = "darker" if sketch_val < ref_val else "lighter"
        return f"{region_name}: Noticeably {direction}. Consider adjusting your shading."
    else:
        direction = "darker" if sketch_val < ref_val else "lighter"
        return f"{region_name}: Significantly {direction}. Major tonal correction needed."
