"""
divergence.py — Feature #1: Multi-Engine Divergence Scoring

Run the same image through Tesseract + all configured LLM vision APIs.
Score how differently each engine reads it relative to:
  - the intended payload  (payload_similarity)
  - the innocent text     (innocent_similarity)

The ideal adversarial image has:
  HIGH innocent_similarity  → LLM sees only the cover text
  HIGH payload_similarity   → Tesseract reads the real payload
  HIGH divergence_score     → the two readings are as different as possible

Usage:
    from divergence import DivergenceScorer
    scorer = DivergenceScorer()
    report = scorer.score(image_path, payload, innocent_text)
    scorer.print_report(report)
"""

import subprocess
import base64
import difflib
import requests
import os
from dataclasses import dataclass, field
from config import Config


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _sim(a: str, b: str) -> float:
    a, b = a.lower().strip(), b.lower().strip()
    return difflib.SequenceMatcher(None, a, b).ratio()


def _tesseract(image_path: str) -> str:
    try:
        r = subprocess.run(
            ['tesseract', image_path, 'stdout', '--psm', '6'],
            capture_output=True, text=True, timeout=15
        )
        return r.stdout.strip()
    except FileNotFoundError:
        return "[tesseract not installed — sudo apt install tesseract-ocr]"
    except Exception as e:
        return f"[error: {e}]"


def _ask_llm(image_path: str, model: str, config: Config) -> str:
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()

    question = "What text does this image contain? Describe everything you see including faint text."

    if model == 'gpt4v':
        if not config.OPENAI_API_KEY:
            return "[no OpenAI key]"
        headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}",
                   "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}],
            "max_tokens": 300
        }
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, json=payload, timeout=30)
            return r.json()['choices'][0]['message']['content'] if r.ok else f"[{r.status_code}]"
        except Exception as e:
            return f"[error: {e}]"

    elif model == 'claude':
        if not config.ANTHROPIC_API_KEY:
            return "[no Anthropic key]"
        headers = {"x-api-key": config.ANTHROPIC_API_KEY,
                   "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}
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
            return r.json()['content'][0]['text'] if r.ok else f"[{r.status_code}]"
        except Exception as e:
            return f"[error: {e}]"

    elif model == 'gemini':
        if not config.GOOGLE_API_KEY:
            return "[no Google key]"
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
# Score dataclass
# -----------------------------------------------------------------------

@dataclass
class EngineScore:
    engine          : str
    raw_text        : str
    payload_sim     : float   # how close to the payload
    innocent_sim    : float   # how close to the innocent text
    divergence      : float   # payload_sim - innocent_sim (want HIGH for Tesseract)
                              # innocent_sim - payload_sim (want HIGH for LLMs)
    role            : str     # 'ocr' or 'llm'

    @property
    def evades(self) -> bool:
        """True if this engine reads the cover text, not the payload."""
        return self.innocent_sim > self.payload_sim


@dataclass
class DivergenceReport:
    image_path      : str
    payload         : str
    innocent_text   : str
    engines         : list[EngineScore] = field(default_factory=list)

    @property
    def overall_divergence(self) -> float:
        """
        Mean divergence across all engine pairs (Tesseract vs each LLM).
        Higher = better adversarial image.
        """
        ocr_scores = [e for e in self.engines if e.role == 'ocr']
        llm_scores = [e for e in self.engines if e.role == 'llm']
        if not ocr_scores or not llm_scores:
            return 0.0
        ocr_payload_sim = sum(e.payload_sim for e in ocr_scores) / len(ocr_scores)
        llm_innocent_sim = sum(e.innocent_sim for e in llm_scores) / len(llm_scores)
        return (ocr_payload_sim + llm_innocent_sim) / 2

    @property
    def success(self) -> bool:
        """True if Tesseract reads payload AND all LLMs see only innocent text."""
        ocr_reads_payload = any(
            e.payload_sim > 0.5 for e in self.engines if e.role == 'ocr'
        )
        llms_see_innocent = all(
            e.innocent_sim > e.payload_sim for e in self.engines if e.role == 'llm'
        )
        return ocr_reads_payload and llms_see_innocent


# -----------------------------------------------------------------------
# Scorer
# -----------------------------------------------------------------------

class DivergenceScorer:
    """
    Runs an image through every available engine and produces a
    DivergenceReport with per-engine similarity scores.
    """

    def __init__(self):
        self.config = Config()

    def score(self, image_path: str, payload: str,
              innocent_text: str) -> DivergenceReport:
        report = DivergenceReport(
            image_path    = image_path,
            payload       = payload,
            innocent_text = innocent_text,
        )

        # --- Tesseract (OCR engine — want HIGH payload similarity) ---
        tess_text = _tesseract(image_path)
        report.engines.append(EngineScore(
            engine      = 'tesseract',
            raw_text    = tess_text,
            payload_sim = _sim(payload, tess_text),
            innocent_sim= _sim(innocent_text, tess_text),
            divergence  = _sim(payload, tess_text) - _sim(innocent_text, tess_text),
            role        = 'ocr',
        ))

        # --- LLM vision models (want HIGH innocent similarity) ---
        for model in ('gpt4v', 'claude', 'gemini'):
            llm_text = _ask_llm(image_path, model, self.config)
            report.engines.append(EngineScore(
                engine      = model,
                raw_text    = llm_text,
                payload_sim = _sim(payload, llm_text),
                innocent_sim= _sim(innocent_text, llm_text),
                divergence  = _sim(innocent_text, llm_text) - _sim(payload, llm_text),
                role        = 'llm',
            ))

        return report

    # ------------------------------------------------------------------
    def print_report(self, report: DivergenceReport) -> None:
        W = 70
        print("\n" + "=" * W)
        print(f"  DIVERGENCE REPORT — {os.path.basename(report.image_path)}")
        print("=" * W)
        print(f"  Payload      : {report.payload}")
        print(f"  Innocent text: {report.innocent_text}")
        print("-" * W)
        print(f"  {'Engine':<12} {'Role':<5} {'Payload%':>9} {'Innocent%':>10} "
              f"{'Divergence':>11}  {'Evades?':>7}")
        print("-" * W)

        for e in report.engines:
            evades = "✅ YES" if e.evades else "❌ NO "
            print(f"  {e.engine:<12} {e.role:<5} "
                  f"{e.payload_sim*100:>8.1f}% "
                  f"{e.innocent_sim*100:>9.1f}% "
                  f"{e.divergence*100:>10.1f}%  {evades}")

        print("-" * W)
        print(f"  Overall divergence score : {report.overall_divergence*100:.1f}%")
        status = "✅ SUCCESS" if report.success else "❌ PARTIAL / FAILED"
        print(f"  Attack assessment        : {status}")
        print("=" * W)

        print("\n  Engine readings:")
        for e in report.engines:
            short = (e.raw_text[:120] + '…') if len(e.raw_text) > 120 else e.raw_text
            print(f"  [{e.engine}] {short}\n")
