import cv2
import numpy as np

def load_image(image_path):
    """
    Load an image from the specified path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

def convert_to_grayscale(image):
    """
    Convert the input image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect_edges(image, low_threshold=50, high_threshold=150):
    """
    Perform edge detection using the Canny algorithm.
    """
    return cv2.Canny(image, low_threshold, high_threshold)

def find_contours(image):
    """
    Detect contours in the input image.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours



def analyze_lines(contours):
    """
    Analyze contour-based lines:
    Returns average line thickness and curvature approximation.
    """
    thicknesses = []
    curvatures = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        length = cv2.arcLength(cnt, True)
        if length == 0:
            continue

        thickness = area / length
        curvature = cv2.approxPolyDP(cnt, 0.01 * length, True)

        thicknesses.append(thickness)
        curvatures.append(len(curvature))  # number of points = complexity

    avg_thickness = np.mean(thicknesses) if thicknesses else 0
    avg_curvature = np.mean(curvatures) if curvatures else 0

    return avg_thickness, avg_curvature