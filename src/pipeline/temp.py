import os
import time
import asyncio
from config import config
from pprint import pprint
from pathlib import Path
from src.pipeline.extraction.layoutLM_v3 import LayoutLMv3
from src.pipeline.extraction.gemini import Gemini
from src.pipeline.ocr.OCREngine import OCREngine
from src.pipeline import utils
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor


async def main():
    load_dotenv()
    thread_pool_executor = ThreadPoolExecutor(max_workers=config["max_workers"])
    file_1 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-3.jpeg")
    file_2 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-2.jpeg")
    file_3 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-7.png")
    file_4 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-9.jpg")
    files = [file_1,  file_2, file_3, file_4]
    start_time = time.time()
    ocr_engine = OCREngine(files, thread_pool_executor)
    data = await ocr_engine.perform_ocr()
    '''
    layoutlm_model = LayoutLMv3(thread_pool_executor)
    preds = layoutlm_model.predict_stream(data)
    end_time = time.time()
    pprint(preds)
    '''

    # pprint(data)
    gemini = Gemini(os.getenv("GEMINI_API_KEY"))
    system_prompt = utils.get_system_prompt()
    for item in data:
        ocr_text = ocr_engine.image_to_data_to_string(item['ocr'])
        user_prompt = utils.get_formated_prompt(ocr_data=ocr_text)
        gemini_output = gemini.predict(user_prompt, system_prompt)
        pprint(gemini_output)
    end_time = time.time()
    print("Time taken: ", round(end_time-start_time, 2))
    print("Time taken: ", (end_time - start_time))

if __name__ == "__main__":
    asyncio.run(main())