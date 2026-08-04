import cv2
import os
import numpy as np


input_folder = "dataset/catt"
output_folder = "dataset/catte"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    # Save original image
    cv2.imwrite(os.path.join(output_folder, f"{filename}_original.png"), img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_gray.png"), gray)

    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_blurred.png"), blurred)

    edges = cv2.Canny(blurred, threshold1=80, threshold2=150)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_edges.png"), edges)

    inverted = cv2.bitwise_not(edges)
    final_path = os.path.join(output_folder, f"{filename}_sketch.png")
    cv2.imwrite(final_path, inverted)
