import time
from  google import genai
from loguru import logger
from google.genai import types
from config import config

class Gemini:
    def __init__(self, api_key: str, **kwargs):
        self.client = genai.Client(api_key=api_key)
        logger.info("Connection with Gemini established successfully!")
        self.json_schema = {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": ["string", "null"]
                },
                "company_address": {
                    "type": ["string", "null"]
                },
                "invoice_date": {
                    "type": ["string", "null"],
                    "description": "Invoice date in YYYY-MM-DD format"
                },
                "total_amount": {
                    "type": ["number", "null"]
                }
            },
            "required": {
                "company_name",
                "company_address",
                "invoice_date",
                "total_amount"
            }
        }



    def predict(self, user_prompt:str, system_prompt: str, **kwargs):
        gemini_config = config["gemini_config"]
        for model in gemini_config["model_list"]:
            for attempt in range(1, gemini_config["max_attempts_per_model"] + 1):
                response = self.client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=gemini_config["temperature"],
                        top_p=gemini_config["temp_p"],
                        top_k=gemini_config["top_k"],
                        candidate_count=gemini_config["candidate_count"],
                        max_output_tokens=gemini_config["max_output_tokens"],
                        response_mime_type="application/json",
                        response_json_schema=self.json_schema,
                    )
                )
                output = response.parsed
                if output and dict(output):
                    return dict(output)
                else:
                    logger.warning(f"{model} attempt {attempt} failed! Retrying in {gemini_config["delay_between_consecutive_queries"]} seconds ...")
                    time.sleep(gemini_config["delay_between_consecutive_queries"])
                    continue
        raise RuntimeError("Gemini failed completely!")



