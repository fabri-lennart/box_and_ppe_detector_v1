from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

# Model
MODEL_NAME = "yolov8n.pt"
MODEL_PATH = MODELS_DIR / MODEL_NAME
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

# Window
WINDOW_NAME = "Box & PPE Detector v1"
EXIT_KEY = ord("q")

# Classes of interest for phase 1 (pretrained COCO model).
# In future phases this will be defined by a custom trained model.
CLASSES_OF_INTEREST: set[str] = {
    "person",
    "backpack",
    "handbag",
    "suitcase",
}