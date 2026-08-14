from path_config import PROMPTS_DIR


def get_formated_prompt(ocr_data: str):
    user_prompt = None
    path = PROMPTS_DIR / "user_prompt.txt"
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            user_prompt = file.read()
        if user_prompt:
            user_prompt = user_prompt.format(ocr_data=ocr_data)
            return user_prompt
        else:
            raise ValueError("Unable to read prompt from file 'user_prompt.txt'")
    else:
        raise FileNotFoundError(f"'user_prompt.txt' file not found at location {path}")


def get_system_prompt():
    system_prompt = None
    path = PROMPTS_DIR / "system_prompt.txt"
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            system_prompt = file.read()
        if system_prompt:
            return system_prompt
        else:
            raise ValueError("Unable to read prompt from file 'system_prompt.txt'")
    else:
        raise FileNotFoundError(f"'system_prompt.txt' file not found at location {path}")