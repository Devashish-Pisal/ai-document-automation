from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATASET_PATH = DATA_DIR / "raw" / "SROIE2019" / "train"
TEST_DATASET_PATH = DATA_DIR / "raw" / "SROIE2019" / "test"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
PROMPTS_DIR = DATA_DIR / "prompts"

MODEL_DATA_PATH = PROJECT_ROOT / "model_data"
