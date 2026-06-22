import time
import asyncio
from pprint import pprint
from src.pipeline.ocr.OCREngine import OCREngine
from pathlib import Path

async def main():
    file_1 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-1.jpeg")
    file_2 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-2.jpeg")
    file_3 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-7.png")
    file_4 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-9.jpg")
    file_5 = Path("E:\(_Coding_Data_)\(_Github_Repositories_)\\ai-invoice-automation\samples\sample-10.pdf")
    files = [file_1, file_2, file_3, file_4, file_5]
    start_time = time.time()
    ocr_engine = OCREngine(files)
    data = await ocr_engine.perform_ocr()
    end_time = time.time()
    pprint(data)
    print("Time taken: ", round(end_time-start_time, 2) )
    print("Time taken: ", (end_time - start_time))

if __name__ == "__main__":
    asyncio.run(main())