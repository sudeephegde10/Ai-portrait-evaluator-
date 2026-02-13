# 🎨 AI Portrait Evaluator

An AI-powered web application that evaluates the accuracy of portrait sketches and paintings by comparing them against reference photos. Built for research and educational purposes.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Flask](https://img.shields.io/badge/Flask-3.1-green) ![OpenCV](https://img.shields.io/badge/OpenCV-4.10-orange) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-red)

## Overview

Upload a reference portrait photograph and your sketch/painting. The system analyzes three dimensions of accuracy:

| Dimension | Weight | Method |
|-----------|--------|--------|
| **Structural** | 40% | Canny edge detection + 16×16 grid density comparison + SSIM |
| **Proportion** | 30% | MediaPipe FaceMesh landmark ratios (eye spacing, nose length, etc.) |
| **Tonal** | 30% | Region-based mean intensity comparison (forehead, eyes, cheeks, etc.) |

**Final Score = 0.4 × Structural + 0.3 × Proportion + 0.3 × Tonal**

## Installation

```bash
# Clone or navigate to the project directory
cd ai_portrait_evaluator

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Running Locally

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## How It Works

### Structural Score (Canny + SSIM)
Both images are aligned using facial landmarks, converted to grayscale, and edge-detected with Canny. The edge maps are compared via:
- **Grid density comparison** — 16×16 grid blocks, comparing edge pixel ratios
- **SSIM (Structural Similarity Index)** — measures perceptual similarity between edge maps

### Proportion Score (Landmark Ratios)
MediaPipe FaceMesh extracts 468 facial landmarks. Key geometric ratios are computed (eye spacing, nose length, mouth width, etc.) and compared between reference and sketch.

### Tonal Score (Region Brightness)
The face is divided into regions (forehead, eyes, nose, cheeks, chin) using landmarks. Mean pixel intensity is compared per region to evaluate light/shadow accuracy.

### Heatmap Visualization
A pixel-level absolute difference map is generated, colorized with JET colormap, and overlaid on the sketch. Red = high error, blue = good match.

## Research Data

Each evaluation is logged to `results.csv` with columns:
```
timestamp, reference_image, sketch_image, structural_score, proportion_score, tonal_score, final_score
```

## Limitations

> ⚠️ This system evaluates **structural similarity**, **geometric proportion**, and **tonal alignment** only.
>
> It does **NOT** evaluate:
> - Artistic style or technique
> - Creativity or expression
> - Intentional stylistic exaggeration
> - Color accuracy (analysis is grayscale-based)

## Deployment (Render)

1. Push to a Git repository
2. Create a new **Web Service** on [Render](https://render.com)
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Deploy

## Tech Stack

- **Backend**: Python, Flask, OpenCV, MediaPipe, NumPy, scikit-image, rembg
- **Frontend**: HTML, CSS, JavaScript, Canvas API, Chart.js
- **Deployment**: Gunicorn

## License

For research and educational purposes.
