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
        if not ocr:
            logger.error(f"OCR data for file {file_path} is 'None'") # Check for None ocr values when doing model inference and skip those images
        return {
            "file": file_path,
            "ocr": ocr,
        }





