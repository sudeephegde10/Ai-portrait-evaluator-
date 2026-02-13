"""
app.py — Flask Application for AI Portrait Evaluator

Main entry point. Provides routes for:
    - GET  /                  — Serve the single-page UI
    - POST /analyze           — Accept two images, run analysis pipeline, return JSON
    - GET  /uploads/<filename> — Serve uploaded/generated image files

Also handles CSV logging of evaluation results for research purposes.
"""

import os
import csv
import uuid
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

# Import analysis modules
from utils.preprocess import preprocess_pair
from utils.structural import analyze_structure
from utils.proportion import analyze_proportion
from utils.tonal import analyze_tonal
from utils.scoring import (compute_final_score, generate_heatmap,
                           generate_feedback, get_grade)

# ---------------------------------------------------------------------------
# Flask App Configuration
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size

# Upload folder setup
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# CSV log file for research data
CSV_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results.csv")

# Allowed image extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "tiff"}


def allowed_file(filename):
    """Check if file has an allowed image extension."""
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main single-page application."""
    return render_template("index.html")


@app.route("/uploads/<filename>")
def serve_upload(filename):
    """Serve files from the uploads directory (images, heatmaps)."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main analysis endpoint.

    Expects multipart form data with:
        - reference: reference portrait photo file
        - sketch: user's sketch/painting file

    Returns JSON with:
        - structural_score, proportion_score, tonal_score, final_score
        - grade
        - proportion_details, tonal_details
        - feedback (list of strings)
        - heatmap_url
        - ref_image_url, sketch_image_url
    """
    # --- Validate uploaded files ---
    if "reference" not in request.files or "sketch" not in request.files:
        return jsonify({
            "error": "Please upload both a reference photo and a sketch."
        }), 400

    ref_file = request.files["reference"]
    sketch_file = request.files["sketch"]

    if ref_file.filename == "" or sketch_file.filename == "":
        return jsonify({
            "error": "Please select both files before analyzing."
        }), 400

    if not allowed_file(ref_file.filename) or not allowed_file(sketch_file.filename):
        return jsonify({
            "error": "Invalid file type. Please upload PNG, JPG, BMP, or WebP images."
        }), 400

    # --- Save uploaded files with unique names ---
    session_id = str(uuid.uuid4())[:8]
    ref_ext = ref_file.filename.rsplit(".", 1)[1].lower()
    sketch_ext = sketch_file.filename.rsplit(".", 1)[1].lower()

    ref_filename = f"{session_id}_ref.{ref_ext}"
    sketch_filename = f"{session_id}_sketch.{sketch_ext}"

    ref_path = os.path.join(UPLOAD_FOLDER, ref_filename)
    sketch_path = os.path.join(UPLOAD_FOLDER, sketch_filename)

    ref_file.save(ref_path)
    sketch_file.save(sketch_path)

    # --- Run preprocessing pipeline ---
    try:
        prep = preprocess_pair(ref_path, sketch_path)
    except Exception as e:
        return jsonify({
            "error": f"Preprocessing failed: {str(e)}"
        }), 500

    if prep["error"]:
        return jsonify({"error": prep["error"]}), 400

    # --- Run analysis modules ---
    try:
        # Structural analysis
        structural_result = analyze_structure(prep["ref_gray"],
                                              prep["sketch_gray"])

        # Proportion analysis
        proportion_result = analyze_proportion(prep["ref_landmarks"],
                                               prep["sketch_landmarks"])

        # Tonal analysis
        tonal_result = analyze_tonal(prep["ref_gray"], prep["sketch_gray"],
                                     prep["ref_landmarks"],
                                     prep["sketch_landmarks"])

        # Final score
        final_score = compute_final_score(
            structural_result["score"],
            proportion_result["score"],
            tonal_result["score"]
        )

        # Generate heatmap
        heatmap_filename = f"{session_id}_heatmap.png"
        heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_filename)
        generate_heatmap(prep["ref_gray"], prep["sketch_gray"], heatmap_path)

        # Generate feedback
        feedback = generate_feedback(structural_result, proportion_result,
                                     tonal_result, final_score)

        # Grade
        grade = get_grade(final_score)

    except Exception as e:
        return jsonify({
            "error": f"Analysis failed: {str(e)}"
        }), 500

    # --- Log results to CSV for research ---
    _log_to_csv(ref_file.filename, sketch_file.filename,
                structural_result["score"],
                proportion_result["score"],
                tonal_result["score"],
                final_score)

    # --- Build response JSON ---
    response = {
        "final_score": final_score,
        "grade": grade,
        "structural_score": structural_result["score"],
        "structural_grid_score": structural_result["grid_score"],
        "structural_ssim_score": structural_result["ssim_score"],
        "proportion_score": proportion_result["score"],
        "proportion_details": proportion_result.get("details", []),
        "tonal_score": tonal_result["score"],
        "tonal_details": tonal_result.get("details", []),
        "feedback": feedback,
        "heatmap_url": f"/uploads/{heatmap_filename}",
        "ref_image_url": f"/uploads/{ref_filename}",
        "sketch_image_url": f"/uploads/{sketch_filename}",
    }

    return jsonify(response)


# ---------------------------------------------------------------------------
# CSV Logging (Research Feature)
# ---------------------------------------------------------------------------

def _log_to_csv(ref_name, sketch_name, structural, proportion, tonal, final):
    """
    Append evaluation results to a CSV file for research data analysis.

    Columns: timestamp, reference_image, sketch_image, structural_score,
             proportion_score, tonal_score, final_score
    """
    file_exists = os.path.exists(CSV_LOG_PATH)

    with open(CSV_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not file_exists:
            writer.writerow([
                "timestamp", "reference_image", "sketch_image",
                "structural_score", "proportion_score",
                "tonal_score", "final_score"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            ref_name,
            sketch_name,
            structural,
            proportion,
            tonal,
            final
        ])


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AI Portrait Evaluator")
    print("  http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
