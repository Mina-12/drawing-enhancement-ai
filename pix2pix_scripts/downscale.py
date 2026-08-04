from PIL import Image
import os

input_folder = "dataset/refined_sketches/dog"  # or your full path
output_folder = "dataset/refined_sketches/dog_256"
new_size = (256, 256)  # or (128, 128)

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(new_size, Image.LANCZOS)  # or BICUBIC
        img.save(os.path.join(output_folder, filename))
