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
            "required": [
                "company_name",
                "company_address",
                "invoice_date",
                "total_amount"
            ]
        }



    def predict(self, user_prompt:str, system_prompt: str, **kwargs):
        gemini_config = config["gemini_config"]
        # print(system_prompt)
        # print(user_prompt)
        for model in gemini_config["model_list"]:
            for attempt in range(1, gemini_config["max_attempts_per_model"] + 1):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=gemini_config["temperature"],
                            top_p=gemini_config["top_p"],
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
                except genai.errors.ServerError as e:
                    """
                    GEMINI ERROR EXAMPLE:
                    google.genai.errors.ServerError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}
                    """
                    logger.warning(f"Gemini server error.  Retrying in {gemini_config['delay_between_consecutive_queries']} seconds ...")
                    continue
                except genai.errors.ClientError as e:
                    if e.code == 404:
                        """
                        ERROR EXAMPLE:
                        google.genai.errors.ClientError: 404 NOT_FOUND. {'error': {'code': 404, 'message': 'models/gemini-3-flash-lite-preview is not found for API version v1beta, or is not supported for generateContent. Call ModelService.ListModels to see the list of available models and their supported methods.', 'status': 'NOT_FOUND'}}
                        """
                        logger.warning(f"Model {model} not found. Retrying with next model...")
                        break
                    elif e.code == 429:
                        """
                        ERROR EXAMPLE:
                        google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash-lite\nPlease retry in 16.776793154s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-lite'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '16s'}]}}
                        """
                        logger.warning(f"Daily quota for model {model} is exhausted. Retrying with next model...")
                        break
                except Exception as e:
                    logger.warning(f"UNEXPECTED ERROR OCCURRED FOR MODEL {model}: " + str(e))
                    logger.warning("Retrying immediately...")
        raise RuntimeError("Gemini failed completely!")



