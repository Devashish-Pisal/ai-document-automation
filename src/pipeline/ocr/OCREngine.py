import os
import asyncio
import pytesseract
from PIL import Image
from pathlib import Path
from config import config
from path_config import TEMP_FILES_DIR
from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor

ocr_pool = ProcessPoolExecutor(max_workers=config["max_workers"])

class OCREngine:
    def __init__(self, file_paths:list[Path]):
        self.files_to_process = file_paths
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
            loop.run_in_executor(ocr_pool, self.perform_pytesseract_ocr, file_path)
            for file_path in self.files_to_process
        ]
        result = await asyncio.gather(*tasks)
        return result


    @staticmethod
    def perform_pytesseract_ocr(file_path):
        img = Image.open(file_path)
        text =  pytesseract.image_to_string(img)
        return {
            "file": file_path,
            "text": text,
        }





