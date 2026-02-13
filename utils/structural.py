"""
structural.py — Structural Accuracy Analysis

Evaluates how accurately the sketch reproduces the line structure
of the reference portrait.

Method:
    1. Apply Canny edge detection to both aligned grayscale images
    2. Divide into a 16x16 grid of blocks
    3. Compare edge density per block
    4. Compute SSIM between edge maps
    5. Combine into a structural score (0–100)
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def extract_edges(gray_img, low_thresh=50, high_thresh=150):
    """
    Apply Canny edge detection to a grayscale image.

    Args:
        gray_img: grayscale image (numpy array)
        low_thresh: lower threshold for Canny
        high_thresh: upper threshold for Canny

    Returns:
        Binary edge map (numpy array)
    """
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 1.0)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    return edges


def compute_edge_density(edge_block):
    """
    Calculate the ratio of edge pixels to total pixels in a block.

    Args:
        edge_block: binary edge image block

    Returns:
        Float between 0 and 1
    """
    total_pixels = edge_block.size
    if total_pixels == 0:
        return 0.0
    edge_pixels = np.count_nonzero(edge_block)
    return edge_pixels / total_pixels


def grid_comparison(ref_edges, sketch_edges, grid_size=16):
    """
    Divide edge maps into a grid and compare edge density per block.

    Args:
        ref_edges: reference edge map
        sketch_edges: sketch edge map
        grid_size: number of rows/columns in the grid

    Returns:
        - grid_score: overall similarity score from grid comparison (0–100)
        - block_scores: 2D array of per-block similarity scores (0–1)
    """
    h, w = ref_edges.shape
    block_h = h // grid_size
    block_w = w // grid_size

    block_scores = np.zeros((grid_size, grid_size))
    total_similarity = 0.0
    active_blocks = 0  # Blocks with at least some edge content

    for row in range(grid_size):
        for col in range(grid_size):
            # Extract block from both images
            y1 = row * block_h
            y2 = y1 + block_h
            x1 = col * block_w
            x2 = x1 + block_w

            ref_block = ref_edges[y1:y2, x1:x2]
            sketch_block = sketch_edges[y1:y2, x1:x2]

            # Calculate edge density for each
            ref_density = compute_edge_density(ref_block)
            sketch_density = compute_edge_density(sketch_block)

            # Similarity = 1 - abs difference in density
            max_density = max(ref_density, sketch_density)
            if max_density > 0.01:  # Only count blocks with meaningful edges
                similarity = 1.0 - abs(ref_density - sketch_density) / max_density
                block_scores[row, col] = similarity
                total_similarity += similarity
                active_blocks += 1
            else:
                # Both blocks essentially empty — perfect match
                block_scores[row, col] = 1.0

    # Average similarity across active blocks
    if active_blocks > 0:
        grid_score = (total_similarity / active_blocks) * 100
    else:
        grid_score = 100.0  # No edges to compare

    return grid_score, block_scores


def compute_ssim_score(ref_edges, sketch_edges):
    """
    Compute SSIM between two edge maps.

    Args:
        ref_edges: reference edge map
        sketch_edges: sketch edge map

    Returns:
        SSIM score scaled to 0–100
    """
    score, _ = ssim(ref_edges, sketch_edges, full=True)
    # SSIM ranges from -1 to 1; we clamp and scale to 0–100
    return max(0, score) * 100


def analyze_structure(ref_gray, sketch_gray):
    """
    Full structural analysis pipeline.

    Args:
        ref_gray: aligned reference grayscale image
        sketch_gray: aligned sketch grayscale image

    Returns:
        Dictionary with:
            - score: structural score (0–100)
            - grid_score: grid-based comparison score
            - ssim_score: SSIM-based score
            - block_scores: 2D array of per-block scores
            - ref_edges: reference edge map (for visualization)
            - sketch_edges: sketch edge map (for visualization)
    """
    # Extract edges
    ref_edges = extract_edges(ref_gray)
    sketch_edges = extract_edges(sketch_gray)

    # Grid-based comparison
    grid_score, block_scores = grid_comparison(ref_edges, sketch_edges)

    # SSIM comparison
    ssim_score = compute_ssim_score(ref_edges, sketch_edges)

    # Combine: 60% grid + 40% SSIM
    combined_score = 0.6 * grid_score + 0.4 * ssim_score

    return {
        "score": round(combined_score, 1),
        "grid_score": round(grid_score, 1),
        "ssim_score": round(ssim_score, 1),
        "block_scores": block_scores,
        "ref_edges": ref_edges,
        "sketch_edges": sketch_edges
    }
