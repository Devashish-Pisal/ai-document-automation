from dotenv import load_dotenv
from loguru import logger
import os



HUGGINGFACE_TOKEN = None
GEMINI_API_KEY = None

try:
    load_dotenv()
    HUGGINGFACE_TOKEN = os.getenv("HF_WRITE_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not HUGGINGFACE_TOKEN:
        raise ValueError("HF_WRITE_TOKEN is not set")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

except Exception as e:
    logger.error(f"Unable to load secrets: {e}")
    raise e

