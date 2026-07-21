import torch
import threading
from PIL import Image
from pprint import pprint
from loguru import logger
from config import config
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification



class LayoutLMv3:
    def __init__(self,  thread_pool_executor):
        self.thread_pool_executor = thread_pool_executor
        self.batch_size = config["layout_lm_config"]["batch_size"]
        self.inference_lock = threading.Lock()
        self.processor = LayoutLMv3Processor.from_pretrained("devashish-pisal/layoutlmv3-sroie-token-classification")
        self.model = LayoutLMv3ForTokenClassification.from_pretrained("devashish-pisal/layoutlmv3-sroie-token-classification")
        self.model.eval()
        torch.set_num_threads(config["layout_lm_config"]["torch_threads"])
        logger.info("Fine-tuned LayoutLMv3 model loaded successfully!")

    def predict_batch(self, samples):
        """
        samples:
        [
            {
              "file": "/tmp/a.png",
              "ocr": {...}
            },
            ...
        ]
        """
        images = []
        words_batch = []
        boxes_batch = []
        for item in samples:
            file_path = item["file"]
            ocr_data = item["ocr"]
            with Image.open(file_path) as image:
                image = image.convert("RGB")
                words, boxes = self.create_words_bboxes_from_pytesseract_data(ocr_data,image.width,image.height)
                images.append(image)
                words_batch.append(words)
                boxes_batch.append(boxes)
        encoding = self.processor(
            images,
            words_batch,
            boxes=boxes_batch,
            return_tensors="pt",
            truncation=True,
            padding="longest",
        )
        with self.inference_lock:
            with torch.inference_mode():
                outputs = self.model(**encoding)
        predictions = outputs.logits.argmax(dim=-1)
        # free tensors ASAP
        # del encoding
        del outputs

        #REMOVE LATER==========================
        # decode predictions
        tokens = self.processor.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"][0].cpu().numpy()
        )
        # print result
        id2label = self.model.config.id2label
        pprint(id2label)
        print("\nToken predictions:\n")
        for token, pred in zip(tokens, predictions):
            print(f"{token:15} -> {id2label[pred]}")
        #REMOVE LATER==========================

        return predictions


    def predict_stream(self, samples):
        """
        Automatically batches input
        """
        results = []
        batch = []
        for item in samples:
            batch.append(item)
            if len(batch) == self.batch_size:
                result = self.predict_batch(batch)
                results.extend(result)
                batch.clear()
        # remaining items
        if batch:
            results.extend(self.predict_batch(batch))
        return results


    @staticmethod
    def create_words_bboxes_from_pytesseract_data(ocr_data,img_width,img_height,min_confidence=config["layout_lm_config"]["min_ocr_confidence"]):
        # Create word & bounding box lists from the output of function pytesseract.image_to_data()
        words = []
        boxes = []
        n = len(ocr_data["text"])
        for i in range(n):
            word = ocr_data["text"][i].strip()
            if not word:
                continue
            try:
                conf = float(ocr_data["conf"][i])
            except (ValueError, TypeError):
                continue
            if conf < min_confidence:
                continue
            x = ocr_data["left"][i]
            y = ocr_data["top"][i]
            w = ocr_data["width"][i]
            h = ocr_data["height"][i]
            x0 = int(1000 * x / img_width)
            y0 = int(1000 * y / img_height)
            x1 = int(1000 * (x + w) / img_width)
            y1 = int(1000 * (y + h) / img_height)
            # Clamp to LayoutLM range
            x0 = max(0, min(1000, x0))
            y0 = max(0, min(1000, y0))
            x1 = max(0, min(1000, x1))
            y1 = max(0, min(1000, y1))
            # skip invalid boxes
            if x1 <= x0 or y1 <= y0:
                continue
            words.append(word)
            boxes.append([x0, y0, x1, y1])
        return words, boxes



    @staticmethod
    def _create_words_bboxes_from_pytesseract_string(ocr_string,img_width,img_height):
        # Create word & bounding box lists from the output of function pytesseract.image_to_string()
        words = []
        boxes = []
        for i, word in enumerate(ocr_string["text"]):
            if not word.strip():
                continue
            x = ocr_string["left"][i]
            y = ocr_string["top"][i]
            w = ocr_string["width"][i]
            h = ocr_string["height"][i]
            x0 = int(1000 * x / img_width)
            y0 = int(1000 * y / img_height)
            x1 = int(1000 * (x + w) / img_width)
            y1 = int(1000 * (y + h) / img_height)
            # Clamp to LayoutLM range
            x0 = max(0, min(1000, x0))
            y0 = max(0, min(1000, y0))
            x1 = max(0, min(1000, x1))
            y1 = max(0, min(1000, y1))
            # skip invalid boxes
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([x0, y0, x1, y1])
            words.append(word)
        return words, boxes