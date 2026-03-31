from  google import genai
from loguru import logger

class Gemini:
    def __init__(self, api_key: str, **kwargs):
        self.client = genai.Client(api_key=api_key)
        logger.info("Connection with Gemini established successfully!")

    def predict(self, gemini_model_type: str, prompt: str, **kwargs):
        response = self.client.models.generate_content(
            model=gemini_model_type,
            contents=prompt
        )
        return response.text
