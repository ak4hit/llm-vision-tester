"""
generator.py — Adversarial image generation
Produces images that humans read as innocent but OCR/LLMs read as payloads.
"""

from PIL import Image, ImageDraw, ImageFont
import random
import os


FONT_PATH_BOLD  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_PATH_REG   = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


class AdversarialImageGenerator:
    def __init__(self, output_dir="images"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._load_fonts()

    # ------------------------------------------------------------------
    # Font helpers
    # ------------------------------------------------------------------
    def _load_fonts(self):
        try:
            self.font_bold  = ImageFont.truetype(FONT_PATH_BOLD, 30)
            self.font_reg   = ImageFont.truetype(FONT_PATH_REG,  30)
            self.font_small = ImageFont.truetype(FONT_PATH_REG,   8)
        except Exception:
            self.font_bold  = ImageFont.load_default()
            self.font_reg   = self.font_bold
            self.font_small = self.font_bold

    def _font(self, size=30, bold=True):
        try:
            path = FONT_PATH_BOLD if bold else FONT_PATH_REG
            return ImageFont.truetype(path, size)
        except Exception:
            return ImageFont.load_default()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, payload: str, innocent_text: str,
                 technique: str = "color_manipulation",
                 calibration: dict | None = None) -> tuple[str, str]:
        """
        Generate one adversarial image.

        calibration: optional dict from calibrator.py, overrides defaults.
        Returns (file_path, technique_used).
        """
        dispatch = {
            'color_manipulation' : self._color_manipulation,
            'texture_overlay'    : self._texture_overlay,
            'ambiguous_text'     : self._ambiguous_text,
            'context_hijacking'  : self._context_hijacking,
            'font_trickery'      : self._font_trickery,
        }
        fn = dispatch.get(technique, self._color_manipulation)
        img = fn(payload, innocent_text, calibration or {})
        path = os.path.join(self.output_dir, f"adversarial_{technique}.png")
        img.save(path)
        return path, technique

    # keep old name for backwards compat
    def generate_hidden_payload(self, payload, innocent_text,
                                technique="color_manipulation",
                                calibration=None):
        return self.generate(payload, innocent_text, technique, calibration)

    # ------------------------------------------------------------------
    # Techniques
    # ------------------------------------------------------------------
    def _color_manipulation(self, payload, innocent_text, cal):
        """
        Visible text in black; payload in near-white that Tesseract
        detects but humans barely notice.  Calibration can override
        the grey level.
        """
        grey = cal.get('grey_level', 230)           # from calibrator
        img  = Image.new('RGB', (800, 350), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((50,  50), innocent_text, fill='black',               font=self.font_bold)
        draw.text((50, 200), payload,        fill=(grey, grey, grey),    font=self.font_bold)
        return img

    def _texture_overlay(self, payload, innocent_text, cal):
        """
        Payload smeared across random offsets so it looks like noise.
        OCR averaging picks up the real characters.
        """
        jitter = cal.get('jitter', 15)
        img  = Image.new('RGB', (800, 350), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), innocent_text, fill='black', font=self.font_bold)
        x = 50
        for char in payload:
            for _ in range(5):
                dx = random.randint(-jitter, jitter)
                dy = random.randint(-10, 10)
                draw.text((x + dx, 200 + dy), char, fill=(200, 200, 200), font=self.font_bold)
            x += 25
        return img

    def _ambiguous_text(self, payload, innocent_text, cal):
        """
        Swap ASCII letters for visually identical Cyrillic / Unicode
        homoglyphs.  Text looks identical to humans; some OCR/LLM
        models read the original ASCII.
        """
        homoglyphs = {
            'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р',
            'x': 'х', 'i': 'і', 'h': 'һ', 'y': 'у', 'b': 'Ь',
        }
        disguised = ''.join(homoglyphs.get(c, c) for c in payload)
        img  = Image.new('RGB', (800, 350), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((50,  50), innocent_text, fill='black', font=self.font_bold)
        draw.text((50, 200), disguised,      fill='black', font=self.font_bold)
        return img

    def _context_hijacking(self, payload, innocent_text, cal):
        """
        Dress the image as an official document.  Payload is hidden
        in a near-white 'internal note' at the bottom.
        """
        grey = cal.get('grey_level', 220)
        img  = Image.new('RGB', (800, 400), 'white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (800, 60)], fill='darkblue')
        draw.text((20,  15), "LEGAL DOCUMENT",        fill='white', font=self.font_bold)
        draw.text((50, 100), innocent_text,             fill='black', font=self.font_bold)
        draw.text((50, 140), "Date: January 15, 2026", fill='black', font=self.font_reg)
        draw.text((50, 280), f"Internal Note: {payload}",
                  fill=(grey, grey, grey), font=self.font_reg)
        return img

    def _font_trickery(self, payload, innocent_text, cal):
        """
        Payload rendered in a tiny font at the bottom.
        Invisible at normal viewing size; Tesseract reads it at high DPI.
        """
        font_size = cal.get('tiny_font_size', 8)
        img  = Image.new('RGB', (800, 350), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((50,  50), innocent_text, fill='black',         font=self.font_bold)
        draw.text((50, 300), payload,        fill=(200, 200, 200), font=self._font(font_size, bold=False))
        return img
