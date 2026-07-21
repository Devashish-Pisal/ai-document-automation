import os
import asyncio
import pytesseract
from PIL import Image
from pathlib import Path
from config import config
from loguru import logger
from path_config import TEMP_FILES_DIR
from pdf2image import convert_from_path


class OCREngine:
    def __init__(self, file_paths:list[Path], thread_pool_executor):
        self.files_to_process = file_paths
        self.thread_pool_executor = thread_pool_executor
        self.convert_pdfs_to_images()

    def convert_pdfs_to_images(self):
        img_paths = []
        for file_path in self.files_to_process:
            if file_path.suffix.lower() == ".pdf":
                file_name = file_path.name.removesuffix(".pdf")
                page = convert_from_path(file_path)
                output_file_name = str(TEMP_FILES_DIR/file_name)+ ".png"
                page[0].save(output_file_name, "PNG")
                img_paths.append(output_file_name)
                # os.remove(file_path)
            else:
                img_paths.append(str(file_path))
        self.files_to_process = img_paths


    async def perform_ocr(self):
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(self.thread_pool_executor, self.perform_pytesseract_ocr, file_path)
            for file_path in self.files_to_process
        ]
        result = await asyncio.gather(*tasks)
        return result


    @staticmethod
    def perform_pytesseract_ocr(file_path):
        ocr = None
        with Image.open(file_path) as img:
            ocr =  pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            """
            EXAMPLE OUTPUT OF image_to_data:
            {
                'level': [1, 2, 3, 4, 5, 5],
                'page_num': [1, 1, 1, 1, 1, 1],
                'block_num': [0, 1, 1, 1, 1, 1],
                'par_num': [0, 0, 1, 1, 1, 1],
                'line_num': [0, 0, 0, 1, 1, 1],
                'word_num': [0, 0, 0, 0, 1, 2],
                'left': [0, 50, 50, 50, 50, 120],
                'top': [0, 100, 100, 100, 100, 100],
                'width': [800, 200, 200, 200, 60, 80],
                'height': [600, 30, 30, 30, 30, 30],
                'conf': [-1, -1, -1, -1, 95, 90],
                'text': ['', '', '', '', 'Hello', 'World']
            }
            """
        if not ocr:
            logger.error(f"OCR data for file {file_path} is 'None'") # Check for None ocr values when doing model inference and skip those images
        return {
            "file": file_path,
            "ocr": ocr,
        }


    @staticmethod
    def image_to_data_to_string(ocr_data):
        n = len(ocr_data["text"])
        paragraphs = {}
        order = []
        for i in range(n):
            text = ocr_data["text"][i].strip()
            if not text:
                continue
            block = ocr_data["block_num"][i]
            par = ocr_data["par_num"][i]
            line = ocr_data["line_num"][i]
            para_key = (block, par)
            line_key = (block, par, line)
            if para_key not in paragraphs:
                paragraphs[para_key] = {}
                order.append(para_key)
            if line_key not in paragraphs[para_key]:
                paragraphs[para_key][line_key] = []
            paragraphs[para_key][line_key].append(text)
        result = []
        for para_key in order:
            lines = paragraphs[para_key]
            sorted_lines = sorted(lines.items(), key=lambda x: x[0][2])
            para_text = "\n".join(
                " ".join(words)
                for _, words in sorted_lines
            )
            result.append(para_text)
        return "\n\n".join(result)



