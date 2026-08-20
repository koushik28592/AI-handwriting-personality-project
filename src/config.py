from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = [
    "Extrovert",
    "Introvert",
    "Optimistic",
    "Pessimistic",
    "Stable_Mindset",
]

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "handwriting_personality_resnet50.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
SPLIT_PATH = ROOT / "reports" / "dataset_split.csv"
REPORT_DIR = ROOT / "reports"

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
