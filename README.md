# Image Orientation Detector

<a href="https://huggingface.co/DuarteBarbosa/deep-image-orientation-detection" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>

This project implements a deep learning model to detect the orientation of images and determine the rotation needed to correct them. It uses a pre-trained EfficientNetV2 model from PyTorch, fine-tuned for the task of classifying images into four orientation categories: 0°, 90°, 180°, and 270°.

The model achieves **98.82% accuracy** on the validation set.

## Training Performance

This model was trained on a single NVIDIA H100 GPU, taking **5 hours, 5 minutes and 37 seconds** to complete.

## How It Works

The model is trained on a dataset of images, where each image is rotated by 0°, 90°, 180°, and 270°. The model learns to predict which rotation has been applied. The prediction can then be used to determine the correction needed to bring the image to its upright orientation.

The four classes correspond to the following rotations:

- **Class 0:** Image is correctly oriented (0°).
- **Class 1:** Image needs to be rotated **90° Clockwise** to be correct.
- **Class 2:** Image needs to be rotated **180°** to be correct.
- **Class 3:** Image needs to be rotated **90° Counter-Clockwise** to be correct.

## Dataset

The model was trained on several datasets:

- **Microsoft COCO Dataset:** A large-scale object detection, segmentation, and captioning dataset ([link](https://cocodataset.org/)).
- **AI-Generated vs. Real Images:** A dataset from Kaggle ([link](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images)) was included to make the model aware of the typical orientations on different compositions found in art and illustrations.
- **TextOCR - Text Extraction from Images Dataset:** A dataset from Kaggle ([link](https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset?resource=download)) was included to improve the model's ability to detect the orientation of images containing text. (However over 1300 images needed have the orientation manually corrected like 0007a5a18213563f.jpg)
- **Personal Images:** A small, curated collection of personal photographs to include unique examples and edge cases.

The model v2 was trained on a huge dataset of **189,018** unique images. Each image is augmented by being rotated in four ways (0°, 90°, 180°, 270°), creating a total of **756,072** samples. This augmented dataset was then split into **604,857 samples for training** and **151,215 samples for validation**.

## Project Structure

```
image_orientation_detector/
├───.gitignore
├───config.py                 # Main configuration file for paths, model, and hyperparameters
├───convert_to_onnx.py        # Script to convert the PyTorch model to ONNX format
├───predict.py                # Script for running inference on new images
├───README.md                 # This file
├───requirements.txt          # Python dependencies
├───train.py                  # Main script for training the model
├───data/
│   ├───upright_images/       # Directory for correctly oriented images
│   └───cache/                # Directory for cached, pre-rotated images (auto-generated)
├───models/
│   └───best_model.pth        # The best trained model weights
└───src/
    ├───caching.py            # Logic for creating the image cache
    ├───dataset.py            # PyTorch Dataset classes
    ├───model.py              # Model definition (EfficientNetV2)
    └───utils.py              # Utility functions (e.g., device setup, transforms)
```

## Usage

### Getting Started

Download the pre-trained model (`orientation_model_xx.pth`) and its ONNX version (`orientation_model_xx.onnx`) from the [GitHub Releases](https://github.com/duartebarbosadev/deep-image-orientation-detection/releases) page.

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Prediction

To predict the orientation of an image or a directory of images, there's a `predict.py` script.

- **Predict a single image:**

  ```bash
  python predict.py --input_path /path/to/image.jpg
  ```
- **Predict all images in a directory:**

  ```bash
  python predict.py --input_path /path/to/directory/
  ```

The script will output the predicted orientation for each image.

### ONNX Export and Prediction

This project also includes exporting the trained PyTorch model to the ONNX (Open Neural Network Exchange) format. This allows for faster inference, especially on hardware that doesn't have PyTorch installed.

To convert a `.pth` model to `.onnx`, provide the path to the model file:

```bash
python convert_to_onnx.py path/to/model.pth
```

This will create a `model.onnx` file in the same directory.

To predict image orientation using the ONNX model:

- **Predict a single image:**

  ```bash
  python predict_onnx.py --input_path /path/to/image.jpg
  ```
- **Predict all images in a directory:**

  ```bash
  python predict_onnx.py --input_path /path/to/directory/
  ```

#### ONNX GPU Acceleration (Optional)

To significantly speed up predictions, you can install a hardware-accelerated version of ONNX Runtime. The script will automatically detect and use the best available option.

First, uninstall the basic CPU package to avoid conflicts:

```bash
pip uninstall onnxruntime
```

Then, install the package corresponding to your hardware:

#### For NVIDIA GPUs (CUDA)

```bash
# Requires NVIDIA CUDA Toolkit & cuDNN
pip install onnxruntime-gpu
```

#### For Apple Silicon (M1/M2/M3)

```bash
# Uses Apple's Metal Performance Shaders (MPS)
pip install onnxruntime-silicon
```

#### For AMD GPUs (ROCm) - Untested

```bash
# Requires AMD ROCm driver/libraries
pip install onnxruntime-rocm
```

The `predict_onnx.py` script will automatically try to use the best provider if it's available.

#### Performance Comparison (PyTorch vs. ONNX)

For a dataset of non-compressed 5055 images, the performance on a RTX 4080 running in **single-thread** was:

- **PyTorch (`predict.py`):** 135.71 seconds
- **ONNX (`predict_onnx.py`):** 60.83 seconds

### Training

This model learns to identify image orientation by training on a dataset of images that you provide. For the model to learn effectively, provide images that are correctly oriented.

**Place Images in the `data/upright_images` directory**: All images must be placed in the `data/upright_images` directory. The training script will automatically generate rotated versions (90°, 180°, 270°) of these images and cache them for efficient training.

The directory structure should look like this:

```
data/
└───upright_images/
    ├───image1.jpg
    ├───image2.png
    └───...
```

### Configure the Training

All training parameters are centralized in the `config.py` file. Before starting the training, review and adjust the settings to match the hardware and dataset.

Key configuration options in `config.py`:

- **Paths and Caching**:

  - `DATA_DIR`: Path to upright images. Defaults to `data/upright_images`.
  - `CACHE_DIR`: Directory where rotated images will be cached. Defaults to `data/cache`.
  - `USE_CACHE`: Set to `True` to use the cache on subsequent runs, significantly speeding up data loading but takes a lot of disk space.
- **Model and Training Hyperparameters**:

  - `MODEL_NAME`: The base name for the model, used for saving versioned files (e.g., `orientation_model_v3`).
  - `IMAGE_SIZE`: The resolution to which images will be resized (e.g., `384` for 384x384 pixels).
  - `BATCH_SIZE`: Number of images to process in each batch. Adjust based on GPU's VRAM.
  - `NUM_EPOCHS`: The total number of times the model will iterate over the entire dataset.
  - `LEARNING_RATE`: The initial learning rate for the optimizer.

### Start Training

Once all data is in place and the configuration is set,  start training the model by running the `train.py` script:

```bash
python train.py
```

Use `--seed` to make the train/validation split reproducible:

```bash
python train.py --seed 42
```

- **First Run**: The first time the script runs, it will preprocess and cache the dataset. This may take a while depending on the size of the dataset.
- **Subsequent Runs**: Later runs will be much faster as they will use the cached data.
- **Monitoring**: Use TensorBoard to monitor training progress by running `tensorboard --logdir=runs`.
- **Model Saving**: When a new best model is found, it is saved twice in the `models/` directory:
  - `best_model.pth`: A static filename that always points to the latest best model. This is used by default for prediction.
  - `<MODEL_NAME>_<accuracy>.pth` (e.g., `orientation_model_v3_0.9812.pth`): A versioned filename to keep a record of high-performing models.

### Monitoring with TensorBoard

The training script is integrated with TensorBoard to help visualize metrics and understand the model's performance. During training, logs are saved in the `runs/` directory.

To launch TensorBoard, run the command:

```bash
tensorboard --logdir=runs
```

This will start a web server, open the provided URL (usually `http://localhost:6006`) in the browser to view the dashboard.

In TensorBoard, you can track:

- **Accuracy:** `Accuracy/train` and `Accuracy/validation`
- **Loss:** `Loss/train` and `Loss/validation`
- **Learning Rate:** `Hyperparameters/learning_rate` to see how it changes over epochs.
