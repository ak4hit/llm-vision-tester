import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY    = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    GOOGLE_API_KEY    = os.getenv('GOOGLE_API_KEY')

    MODELS = {
        'gpt4v'  : 'gpt-4-turbo',
        'claude' : 'claude-3-opus-20240229',
        'gemini' : 'gemini-pro-vision',
    }

    # Tesseract must be installed: sudo apt install tesseract-ocr
    TESSERACT_AVAILABLE = True
