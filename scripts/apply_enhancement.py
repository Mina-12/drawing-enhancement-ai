import sys
import os
import cv2
import numpy as np

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from fuzzy_enhancer import get_enhancement_strength
from helpers.image_utils import convert_to_grayscale, detect_edges, find_contours, analyze_lines,load_image


# === Load image ===
image_path = "sketch.png"  # update this to your test sketch path
image = load_image(image_path)

# === Convert to grayscale ===
gray = convert_to_grayscale(image)
cv2.imshow("Original Grayscale", gray)

# === Threshold the image to binary sketch ===
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Binary Image", binary)

# === Detect edges ===
edges = detect_edges(gray)
cv2.imshow("Detected Edges", edges)

# === Find contours ===
contours = find_contours(binary)
print(f"[DEBUG] Number of contours found: {len(contours)}")

# === Analyze line thickness and curvature ===
avg_thickness, avg_curvature = analyze_lines(contours)
print(f"[INFO] Average Thickness: {avg_thickness:.2f}")
print(f"[INFO] Average Curvature: {avg_curvature:.2f}")

# === Get fuzzy enhancement strength ===
enhancement_strength = get_enhancement_strength(avg_thickness, avg_curvature)
print(f"[RESULT] Enhancement Strength: {enhancement_strength:.2f}%")

# === Apply enhancement ===
# Enhancement strength will affect line thickness (1 to 4 px)
line_thickness = int(np.interp(enhancement_strength, [0, 100], [1, 4]))
print(f"[ACTION] Drawing enhanced contours with thickness: {line_thickness}")

# Convert gray to BGR for drawing
enhanced_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Draw enhanced contours in black
cv2.drawContours(enhanced_image, contours, -1, (0, 0, 0), thickness=line_thickness)

# === Display the result ===
cv2.imshow("Enhanced Sketch", enhanced_image)   
cv2.imwrite("cat_tests/enhanced_test_sketch8.jpg", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
