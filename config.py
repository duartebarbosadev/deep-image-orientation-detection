USE_CACHE = False  # This is much faster for training, but requires disk space.
CACHE_DIR = "data/cache"

# --- Dataloader and Preprocessing ---
DATA_DIR = "data/upright_images"
IMAGE_SIZE = 384  # EfficientNetV2-S was pretrained at 384x384 — keep at 384 for best accuracy
BATCH_SIZE = 64  # Or More (eg. 512), depending on your GPU memory
NUM_WORKERS = 4  # Or More (eg. 16), depending on your CPU cores
DATALOADER_PREFETCH_FACTOR = 2  # Increase if the GPU is waiting on data; decrease if RAM usage is too high. (increase if high data_ms in the logs)

# --- Model Configuration ---
MODEL_SAVE_DIR = "models"
MODEL_NAME = "orientation_model_v7"
NUM_CLASSES = 4  # 0°, 90°, 180°, 270°

# The model is trained to predict the rotation that was APPLIED to an upright image.
# 0: 0°, 1: 90° CW, 2: 180°, 3: 90° CCW
ROTATIONS = {0: 0, 1: 90, 2: 180, 3: 270}

# --- Training Hyperparameters ---
LEARNING_RATE = 0.0001
NUM_EPOCHS = 25
NUM_BLOCKS_TO_UNFREEZE = 2

# --- Prediction Settings ---
# A dictionary to map class indices to the corrective action.
# This is the INVERSE of the rotation applied during training data generation.
CLASS_MAP = {
    0: "Image is correctly oriented (0°).",
    1: "Image needs to be rotated 90° Clockwise to be correct.",
    2: "Image needs to be rotated 180° to be correct.",
    3: "Image needs to be rotated 90° Counter-Clockwise to be correct.",
}
