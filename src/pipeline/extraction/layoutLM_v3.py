from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from loguru import logger
from path_config import MODEL_DATA_PATH, PROJECT_ROOT
import torch
import pytesseract
from PIL import Image
import torch


class LayoutLMv3:
    def __init__(self,  **kwargs):
        self.processor = LayoutLMv3Processor.from_pretrained("devashish-pisal/layoutlmv3-sroie-token-classification")
        self.model = LayoutLMv3ForTokenClassification.from_pretrained("devashish-pisal/layoutlmv3-sroie-token-classification")
        self.model.eval()
        logger.info("Fine-tuned LayoutLMv3 model loaded successfully!")

    def predict(self, image: Image, ocr_data: str):
        width, height = image.size
        words, bboxes = self._create_words_bboxes(ocr_data, width, height)
        assert len(words) == len(bboxes)
        encoding = self.processor(
            image,
            words,
            boxes=bboxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        with torch.no_grad:
            outputs = self.model(**encoding)
        predictions = torch.argmax(outputs.logits)



    def _create_words_bboxes(self, ocr_data, img_width, img_height):
        words = []
        boxes = []
        for i, word in enumerate(ocr_data["text"]):
            if word.strip() == "":
                continue
            x, y, w, h = (
                ocr_data["left"][i],
                ocr_data["top"][i],
                ocr_data["width"][i],
                ocr_data["height"][i],
            )
            # normalize to 0–1000
            box = [
                int(1000 * x / img_width),
                int(1000 * y / img_height),
                int(1000 * (x + w) / img_width),
                int(1000 * (y + h) / img_height),
            ]
            words.append(word)
            boxes.append(box)
        return words, boxes
