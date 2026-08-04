import torch
from torchvision import transforms
from PIL import Image
from models.generator import UNetGenerator
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load generator
gen = UNetGenerator(input_channels=1, output_channels=3).to(device)
gen.load_state_dict(torch.load("checkpoints2/gen_epoch100.pth", map_location=device))
gen.eval()

# Define transform for sketches
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Input sketch path
sketch_path = "cat_tests/test_sketch8.jpg"
img = Image.open(sketch_path).convert("L")
img_tensor = transform(img).unsqueeze(0).to(device)

# Generate
with torch.no_grad():
    fake_img = gen(img_tensor).squeeze(0).cpu()

# Denormalize
fake_img = (fake_img * 0.5 + 0.5).clamp(0, 1)

# Save result
save_path = "cat_tests/generated.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
transforms.ToPILImage()(fake_img).save(save_path)
print(f"Generated image saved to {save_path}")
