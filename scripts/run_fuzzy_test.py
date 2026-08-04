import cv2
import numpy as np
from scripts.fuzzy_enhancer import get_enhancement_strength
from helpers.image_utils import load_image, convert_to_grayscale, detect_edges, find_contours, analyze_lines

# === Load sketch image ===
image_path = "cat_tests/test_sketch5.jpg"  # <-- Replace with your test sketch image path
image = load_image(image_path)

# === Preprocess image ===
gray = convert_to_grayscale(image)
edges = detect_edges(gray)
contours = find_contours(edges)

# === Analyze sketch features ===
avg_thickness, avg_curvature = analyze_lines(contours)
print(f"[INFO] Thickness: {avg_thickness:.2f}, Curvature: {avg_curvature:.2f}")

# === Fuzzy enhancement strength ===
enhancement_score = get_enhancement_strength(avg_thickness, avg_curvature)
print(f"[RESULT] Enhancement Strength: {enhancement_score:.2f}%")
