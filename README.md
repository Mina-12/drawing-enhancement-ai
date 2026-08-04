## Author

**Mina Remon**

Graduation Project – German University in Cairo

Computer Engineering

# Drawing Enhancement AI

Drawing Enhancement AI is a deep learning application that transforms rough hand-drawn sketches of cats and dogs into cleaner, more refined sketches using a Pix2Pix GAN. The project combines computer vision, fuzzy logic, and machine learning to automatically classify and enhance user drawings through an interactive desktop application.

## Features

- Tkinter drawing interface for creating, saving, classifying, and enhancing sketches.
- Classical machine learning drawing classifier with switchable estimators.
- Fuzzy-logic based sketch analysis using contour thickness and curvature.
- Pix2Pix-style GAN pipeline for translating rough grayscale sketches into refined RGB sketch outputs.
- Utility scripts for preprocessing images, resizing datasets, testing checkpoints, and post-processing generated results.

## Architecture

```text
User sketch
   |
   |-- Drawing classifier (`pix2pix_scripts/drawing_classifier.py`)
   |      |-- Predicts class: cat or dog
   |
   |-- Enhancement model
          |-- Cat checkpoint: checkpoints_cats/gen_epoch400.pth
          |-- Dog checkpoint: checkpoints_dogs/gen_epoch500.pth
          |-- U-Net generator (`pix2pix_scripts/models/generator.py`)
          |-- Produces enhanced sketch image

Training pipeline
   |
   |-- Raw/refined paired sketches in `dataset/`
   |-- `Pix2PixDataset`
   |-- U-Net generator + PatchGAN discriminator
   |-- Generated samples in `outputs2/`
   |-- Model checkpoints in `checkpoints2/`
```

## Technologies Used

- Python
- PyTorch and TorchVision
- OpenCV
- Pillow
- NumPy
- scikit-image
- scikit-fuzzy
- scikit-learn
- Matplotlib
- Tkinter

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA acceleration, install the PyTorch build that matches your GPU, CUDA version, and operating system from the official PyTorch installation selector.

## Usage

Run the drawing classifier and enhancement GUI:

```bash
python pix2pix_scripts/drawing_classifier.py
```

Run a pix2pix inference script:

```bash
python pix2pix_scripts/test_pix2pix.py
```

Train the pix2pix model:

```bash
python pix2pix_scripts/train_pix2pix.py
```

Run the fuzzy-logic sketch analysis test:

```bash
python scripts/run_fuzzy_test.py
```

## Dataset Information

The local workspace contains cat and dog sketch/image data under `dataset/` and `dogs_vs_cats/`. These folders are intentionally ignored by Git because datasets can be large, generated, or subject to licensing restrictions.

Expected pix2pix training layout:

```text
dataset/
  raw_sketches/
    dog/
    cat/
  refined_sketches/
    dog/
    cat/
```

Before publishing, add a dataset download link or clear instructions for recreating the dataset. If the images come from third-party sources, include the source, license, and any usage restrictions.

## Model Information

The enhancement model uses:

- `UNetGenerator`: encoder-decoder generator with skip connections.
- `PatchDiscriminator`: PatchGAN-style discriminator over concatenated sketch and real/generated image pairs.
- Losses: adversarial BCE-with-logits loss and L1 reconstruction loss.

Local checkpoint folders are excluded from Git:

- `checkpoints_cats/`
- `checkpoints_dogs/`
- `checkpoints2/`

The current GUI expects these local checkpoints:

- `checkpoints_cats/gen_epoch400.pth`
- `checkpoints_dogs/gen_epoch500.pth`

Because these files are larger than GitHub's normal repository-friendly size, publish them through GitHub Releases, external storage, or Git LFS, then document the download path here.

## Results

Current local outputs include before/after sketch comparisons and generated enhancement images. These are treated as generated artifacts and are ignored by Git.

Suggested public README result format:

| Input Sketch                             | Enhanced Output                           |
| ---------------------------------------- | ----------------------------------------- |
| `docs/screenshots/input-placeholder.png` | `docs/screenshots/output-placeholder.png` |

## Screenshots

Add screenshots before publishing:

- GUI drawing canvas
- Class prediction dialog
- Before/after enhancement example
- Training or inference output sample

Suggested location:

```text
docs/screenshots/
```

## Repository Structure

```text
drawing-enhancement-ai/
  helpers/
    image_utils.py
  pix2pix_scripts/
    models/
      generator.py
      discriminator.py
    dataset.py
    drawing_classifier.py
    image2sketch.py
    train_pix2pix.py
    test_pix2pix.py
    downscale.py
    postprocess.py
    utils.py
  scripts/
    apply_enhancement.py
    fuzzy_enhancer.py
    run_fuzzy_test.py
  requirements.txt
  README.md
  .gitignore
```

## Future Improvements

- Add command-line arguments for checkpoint paths, dataset paths, image paths, and output paths.
- Move generated images into a single `outputs/` directory.
- Add a small curated sample dataset or scripted dataset download.
- Add automated tests for preprocessing utilities and dataset loading.
- Package the source under a single importable module, such as `drawing_enhancement_ai/`.
- Add model cards or release notes for published checkpoints.
- Add reproducible training configuration files.

## License

No license file is currently included. Before publishing publicly, choose a license such as MIT, Apache-2.0, or another license appropriate for your goals and for the dataset/model dependencies.
