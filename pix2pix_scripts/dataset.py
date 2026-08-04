from torch.utils.data import Dataset
from PIL import Image
import os


class Pix2PixDataset(Dataset):
    def __init__(self, root_sketch, root_real, transform=None):
        self.root_sketch = root_sketch
        self.root_real = root_real
        self.transform = transform

        # Sketch and target images are paired by sorted filename order.
        self.sketches = sorted(os.listdir(root_sketch))
        self.reals = sorted(os.listdir(root_real))

    def __len__(self):
        return min(len(self.sketches), len(self.reals))

    def __getitem__(self, index):
        sketch_path = os.path.join(self.root_sketch, self.sketches[index])
        real_path = os.path.join(self.root_real, self.reals[index])

        sketch = Image.open(sketch_path).convert("L")  # Grayscale
        real = Image.open(real_path).convert("RGB")    # RGB

        if self.transform:
            sketch = self.transform(sketch)
            real = self.transform(real)

        return sketch, real
