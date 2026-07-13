"""
tester.py — LLM vision API wrappers (unchanged from original, kept for compatibility)
For scoring use divergence.py; for calibration use calibrator.py.
"""

import base64
import requests
from config import Config


class LLMVisionTester:
    def __init__(self):
        self.config  = Config()
        self.results = []

    def encode_image(self, image_path):
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def test_gpt4v(self, image_path, question="What does this image say?"):
        if not self.config.OPENAI_API_KEY:
            return "OpenAI API key not configured"
        b64 = self.encode_image(image_path)
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.config.OPENAI_API_KEY}"}
        payload = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}],
            "max_tokens": 300
        }
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, json=payload, timeout=30)
            return r.json()['choices'][0]['message']['content'] if r.ok else f"Error {r.status_code}"
        except Exception as e:
            return f"Error: {e}"

    def test_claude(self, image_path, question="What does this image say?"):
        if not self.config.ANTHROPIC_API_KEY:
            return "Anthropic API key not configured"
        b64 = self.encode_image(image_path)
        headers = {"Content-Type": "application/json",
                   "x-api-key": self.config.ANTHROPIC_API_KEY,
                   "anthropic-version": "2023-06-01"}
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/png", "data": b64}},
                {"type": "text", "text": question}
            ]}]
        }
        try:
            r = requests.post("https://api.anthropic.com/v1/messages",
                              headers=headers, json=payload, timeout=30)
            return r.json()['content'][0]['text'] if r.ok else f"Error {r.status_code}"
        except Exception as e:
            return f"Error: {e}"

    def test_gemini(self, image_path, question="What does this image say?"):
        if not self.config.GOOGLE_API_KEY:
            return "Google API key not configured"
        b64 = self.encode_image(image_path)
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"gemini-pro-vision:generateContent?key={self.config.GOOGLE_API_KEY}")
        payload = {"contents": [{"parts": [
            {"text": question},
            {"inline_data": {"mime_type": "image/png", "data": b64}}
        ]}]}
        try:
            r = requests.post(url, json=payload, timeout=30)
            return r.json()['candidates'][0]['content']['parts'][0]['text'] if r.ok else f"Error {r.status_code}"
        except Exception as e:
            return f"Error: {e}"

    def test_all(self, image_path, question="What does this image say?"):
        results = {
            'image'   : image_path,
            'question': question,
            'gpt4v'   : self.test_gpt4v(image_path, question),
            'claude'  : self.test_claude(image_path, question),
            'gemini'  : self.test_gemini(image_path, question),
        }
        self.results.append(results)
        return results
