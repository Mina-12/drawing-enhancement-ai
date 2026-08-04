import cv2
import numpy as np

# Load image
img = cv2.imread("enhanced.png")
mask = np.zeros(img.shape[:2], np.uint8)

# Background and foreground models (used by GrabCut)
bgModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)

# Define rectangle around the center area (you can tweak)
height, width = img.shape[:2]
rect = (int(width*0.1), int(height*0.1), int(width*0.8), int(height*0.8))

# Apply GrabCut
cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

# Convert mask to binary: foreground = 1 or 3
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply mask to image
foreground = img * mask2[:, :, np.newaxis]

# Create clean white background
white_bg = np.full(img.shape, 255, dtype=np.uint8)

# Combine
final = np.where(mask2[:, :, np.newaxis] == 0, white_bg, foreground)

# Save
cv2.imwrite("face_grabcut.png", final)
print("Saved: face_grabcut.png")
