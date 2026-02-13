"""
scoring.py — Final Scoring and Visualization

Handles:
    - Combining structural, proportion, and tonal scores into a final score
    - Generating heatmap visualization of error regions
    - Generating text-based feedback
"""

import cv2
import numpy as np
import os


def compute_final_score(structural_score, proportion_score, tonal_score):
    """
    Compute the weighted final accuracy score.

    Formula: Final = 0.4 * Structural + 0.3 * Proportion + 0.3 * Tonal

    Args:
        structural_score: structural accuracy score (0–100)
        proportion_score: proportion accuracy score (0–100)
        tonal_score: tonal accuracy score (0–100)

    Returns:
        Final weighted score (0–100)
    """
    final = (0.4 * structural_score +
             0.3 * proportion_score +
             0.3 * tonal_score)
    return round(final, 1)


def generate_heatmap(ref_gray, sketch_gray, output_path):
    """
    Generate a heatmap visualization showing error regions.

    Red   = major errors (large difference)
    Green = good match (small difference)

    Args:
        ref_gray: aligned reference grayscale image
        sketch_gray: aligned sketch grayscale image
        output_path: where to save the heatmap image

    Returns:
        Path to the saved heatmap image
    """
    # Compute absolute difference
    difference = cv2.absdiff(ref_gray, sketch_gray)

    # Normalize difference to 0–255 range
    diff_normalized = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)

    # Apply a slight blur for smoother visualization
    diff_blurred = cv2.GaussianBlur(diff_normalized, (15, 15), 0)

    # Apply colormap: COLORMAP_JET gives blue→green→yellow→red
    heatmap = cv2.applyColorMap(diff_blurred.astype(np.uint8), cv2.COLORMAP_JET)

    # Create overlay: blend heatmap with sketch
    sketch_bgr = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(sketch_bgr, 0.4, heatmap, 0.6, 0)

    # Add color legend
    overlay = _add_legend(overlay)

    # Save the heatmap
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)

    return output_path


def _add_legend(img):
    """
    Add a color legend bar to the bottom of the heatmap image.

    Args:
        img: BGR image to add legend to

    Returns:
        Image with legend added
    """
    h, w = img.shape[:2]
    legend_height = 40
    legend = np.zeros((legend_height, w, 3), dtype=np.uint8)

    # Create gradient bar
    bar_y = 10
    bar_h = 20
    bar_margin = 60
    bar_width = w - 2 * bar_margin

    for i in range(bar_width):
        # Map position to value 0–255
        value = int(i * 255 / bar_width)
        color_pixel = np.array([[value]], dtype=np.uint8)
        color_mapped = cv2.applyColorMap(color_pixel, cv2.COLORMAP_JET)
        color = tuple(int(c) for c in color_mapped[0, 0])
        cv2.rectangle(legend, (bar_margin + i, bar_y),
                      (bar_margin + i + 1, bar_y + bar_h), color, -1)

    # Add labels
    cv2.putText(legend, "Good", (5, bar_y + bar_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(legend, "Error", (w - 50, bar_y + bar_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Stack legend below image
    result = np.vstack([img, legend])
    return result


def generate_feedback(structural_result, proportion_result, tonal_result, final_score):
    """
    Generate comprehensive text feedback based on all analysis results.

    Args:
        structural_result: dict from structural analysis
        proportion_result: dict from proportion analysis
        tonal_result: dict from tonal analysis
        final_score: the combined final score

    Returns:
        List of feedback strings
    """
    feedback = []

    # Overall assessment
    if final_score >= 85:
        feedback.append("🌟 Outstanding work! Your sketch is highly accurate.")
    elif final_score >= 70:
        feedback.append("👍 Good job! Your sketch captures the portrait well with minor areas for improvement.")
    elif final_score >= 50:
        feedback.append("📝 Decent effort. Several areas need attention to improve accuracy.")
    else:
        feedback.append("🔄 Keep practicing! Focus on the specific feedback below to improve.")

    # Structural feedback
    s_score = structural_result.get("score", 0)
    if s_score < 60:
        feedback.append("📐 Structure: Your line placement needs significant improvement. "
                        "Focus on the overall shape and contours of the face.")
    elif s_score < 80:
        feedback.append("📐 Structure: Line placement is decent but could be refined. "
                        "Pay attention to subtle contour differences.")
    else:
        feedback.append("📐 Structure: Excellent line accuracy! Your contours match well.")

    # Proportion feedback
    if "details" in proportion_result:
        weak_areas = [d for d in proportion_result["details"]
                      if d.get("accuracy", 100) < 70]
        if weak_areas:
            names = ", ".join(d["name"] for d in weak_areas[:3])
            feedback.append(f"📏 Proportions: Need work on: {names}.")
        else:
            feedback.append("📏 Proportions: Facial geometry is well-captured!")

    # Tonal feedback — include region-specific feedback
    if "details" in tonal_result:
        for detail in tonal_result["details"]:
            if "feedback" in detail:
                feedback.append(f"🎨 {detail['feedback']}")

    return feedback


def get_grade(score):
    """
    Convert a numerical score to a letter grade.

    Args:
        score: numerical score (0–100)

    Returns:
        Letter grade string
    """
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"
