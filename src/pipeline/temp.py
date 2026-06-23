import time
import asyncio
from config import config
from pprint import pprint
from pathlib import Path
from src.pipeline.ocr.OCREngine import OCREngine
from concurrent.futures import ThreadPoolExecutor


async def main():
    thread_pool_executor = ThreadPoolExecutor(max_workers=config["max_workers"])
    file_1 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-1.jpeg")
    file_2 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-2.jpeg")
    file_3 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-7.png")
    file_4 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-9.jpg")
    files = [file_1, file_2, file_3, file_4]
    start_time = time.time()
    ocr_engine = OCREngine(files, thread_pool_executor)
    data = await ocr_engine.perform_ocr()
    end_time = time.time()
    print(data)
    print("Time taken: ", round(end_time-start_time, 2) )
    print("Time taken: ", (end_time - start_time))

if __name__ == "__main__":
    asyncio.run(main())