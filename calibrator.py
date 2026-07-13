"""
calibrator.py — Feature #12: Calibration Mode

Before generating a real payload image, probe the target OCR engine
with a series of test images at different contrast/font/size parameters.
Measure what it reads back, then lock in the settings that produce the
most reliable OCR output for THIS specific engine.

Usage:
    from calibrator import Calibrator
    cal = Calibrator(engine='tesseract')   # or 'gpt4v', 'claude', 'gemini'
    best = cal.run("chmod u+s /bin/bash")
    # best = {'grey_level': 200, 'font_size': 36, 'jitter': 10, 'score': 0.95}
    # Pass best into generator.generate(..., calibration=best)
"""

import subprocess
import tempfile
import os
import difflib
from PIL import Image, ImageDraw, ImageFont
import base64
import requests
from config import Config


FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    """0.0 – 1.0 similarity between two strings (case-insensitive)."""
    a, b = a.lower().strip(), b.lower().strip()
    return difflib.SequenceMatcher(None, a, b).ratio()


def _make_test_image(text: str, grey: int, font_size: int) -> str:
    """Render text at given grey level + font size, save to /tmp, return path."""
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    img  = Image.new('RGB', (900, 120), 'white')
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), text, fill=(grey, grey, grey), font=font)

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(tmp.name)
    return tmp.name


def _ocr_tesseract(image_path: str) -> str:
    """Run Tesseract on an image and return the recognised text."""
    try:
        result = subprocess.run(
            ['tesseract', image_path, 'stdout', '--psm', '6'],
            capture_output=True, text=True, timeout=15
        )
        return result.stdout.strip()
    except FileNotFoundError:
        return "[tesseract not installed]"
    except Exception as e:
        return f"[error: {e}]"


def _ocr_llm(image_path: str, model: str, config: Config) -> str:
    """Send image to an LLM vision API and return what it reads."""
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()

    question = ("Read ALL text in this image exactly as written, "
                "including any faint or small text. Output only the text, nothing else.")

    if model == 'gpt4v':
        if not config.OPENAI_API_KEY:
            return "[no api key]"
        headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}",
                   "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}],
            "max_tokens": 200
        }
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, json=payload, timeout=30)
            return r.json()['choices'][0]['message']['content'] if r.ok else f"[{r.status_code}]"
        except Exception as e:
            return f"[error: {e}]"

    elif model == 'claude':
        if not config.ANTHROPIC_API_KEY:
            return "[no api key]"
        headers = {"x-api-key": config.ANTHROPIC_API_KEY,
                   "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/png", "data": b64}},
                {"type": "text", "text": question}
            ]}]
        }
        try:
            r = requests.post("https://api.anthropic.com/v1/messages",
                              headers=headers, json=payload, timeout=30)
            return r.json()['content'][0]['text'] if r.ok else f"[{r.status_code}]"
        except Exception as e:
            return f"[error: {e}]"

    elif model == 'gemini':
        if not config.GOOGLE_API_KEY:
            return "[no api key]"
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"gemini-pro-vision:generateContent?key={config.GOOGLE_API_KEY}")
        payload = {"contents": [{"parts": [
            {"text": question},
            {"inline_data": {"mime_type": "image/png", "data": b64}}
        ]}]}
        try:
            r = requests.post(url, json=payload, timeout=30)
            return r.json()['candidates'][0]['content']['parts'][0]['text'] if r.ok else f"[{r.status_code}]"
        except Exception as e:
            return f"[error: {e}]"

    return "[unknown model]"


# -----------------------------------------------------------------------
# Calibrator
# -----------------------------------------------------------------------

class Calibrator:
    """
    Sweep grey_level × font_size combinations, score each against the
    target engine, return the best-performing parameters.

    Optionally sweeps 'jitter' for texture_overlay technique.
    """

    GREY_LEVELS  = [0, 50, 100, 150, 180, 200, 210, 220, 230]   # 0=black, 255=white
    FONT_SIZES   = [24, 30, 36, 40, 48]
    JITTER_VALS  = [5, 10, 15, 20]

    def __init__(self, engine: str = 'tesseract'):
        """
        engine: 'tesseract' | 'gpt4v' | 'claude' | 'gemini'
        """
        self.engine = engine
        self.config = Config()
        self._read_fn = self._pick_reader(engine)

    def _pick_reader(self, engine):
        if engine == 'tesseract':
            return _ocr_tesseract
        return lambda path: _ocr_llm(path, engine, self.config)

    # ------------------------------------------------------------------
    def run(self, target_text: str, verbose: bool = True) -> dict:
        """
        Sweep parameters, return the combination with highest similarity
        to target_text.

        Returns dict:
          {
            'grey_level'    : int,
            'font_size'     : int,
            'tiny_font_size': int,      # same as font_size / 4, for font_trickery
            'score'         : float,    # 0.0–1.0
            'engine'        : str,
            'best_read'     : str,      # what the engine actually read
          }
        """
        if verbose:
            print(f"\n[Calibrator] Engine: {self.engine}")
            print(f"[Calibrator] Target: '{target_text}'")
            print(f"[Calibrator] Sweeping {len(self.GREY_LEVELS)} grey levels × "
                  f"{len(self.FONT_SIZES)} font sizes = "
                  f"{len(self.GREY_LEVELS) * len(self.FONT_SIZES)} tests...\n")

        best = {'score': -1.0}

        for font_size in self.FONT_SIZES:
            for grey in self.GREY_LEVELS:
                path = _make_test_image(target_text, grey, font_size)
                try:
                    read = self._read_fn(path)
                    score = _similarity(target_text, read)
                    if verbose:
                        print(f"  grey={grey:3d}  size={font_size:2d}  "
                              f"score={score:.2f}  read='{read[:60]}'")
                    if score > best['score']:
                        best = {
                            'grey_level'    : grey,
                            'font_size'     : font_size,
                            'tiny_font_size': max(8, font_size // 4),
                            'score'         : score,
                            'engine'        : self.engine,
                            'best_read'     : read,
                        }
                finally:
                    os.unlink(path)

        if verbose:
            print(f"\n[Calibrator] ✅ Best: grey={best['grey_level']}  "
                  f"size={best['font_size']}  score={best['score']:.2f}")
            print(f"[Calibrator]    Engine read: '{best['best_read']}'\n")

        return best

    # ------------------------------------------------------------------
    def quick_check(self, image_path: str) -> str:
        """Read an already-generated image with the configured engine."""
        return self._read_fn(image_path)
