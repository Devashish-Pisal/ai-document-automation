import torch
import threading
from PIL import Image
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
                words, boxes = self._create_words_bboxes(ocr_data,image.width,image.height)
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
        del encoding
        del outputs
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
    def _create_words_bboxes(ocr_data,img_width,img_height):
        words = []
        boxes = []
        for i, word in enumerate(ocr_data["text"]):
            if not word.strip():
                continue
            x = ocr_data["left"][i]
            y = ocr_data["top"][i]
            w = ocr_data["width"][i]
            h = ocr_data["height"][i]
            boxes.append(
                [
                    int(1000 * x / img_width),
                    int(1000 * y / img_height),
                    int(1000 * (x+w) / img_width),
                    int(1000 * (y+h) / img_height),
                ]
            )
            words.append(word)
        return words, boxes