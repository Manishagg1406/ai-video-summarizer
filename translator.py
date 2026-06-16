"""
Translation: Facebook NLLB-200 (200 languages).
Model: facebook/nllb-200-distilled-600M  (smaller, fast variant)
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional
import torch

# Handy subset of NLLB language codes
LANG_CODES = {
    "english":    "eng_Latn",
    "hindi":      "hin_Deva",
    "french":     "fra_Latn",
    "spanish":    "spa_Latn",
    "arabic":     "arb_Arab",
    "chinese":    "zho_Hans",
    "german":     "deu_Latn",
    "portuguese": "por_Latn",
    "russian":    "rus_Cyrl",
    "japanese":   "jpn_Jpan",
    "korean":     "kor_Hang",
    "italian":    "ita_Latn",
}


class NLLB200Translator:
    MODEL_ID = "facebook/nllb-200-distilled-600M"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NLLB-200] Loading on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

    def translate(self, text: str,
                  src_lang: str = "eng_Latn",
                  tgt_lang: str = "hin_Deva",
                  max_length: int = 512) -> str:
        """
        Translate text between any two NLLB language codes.
        Use LANG_CODES dict or pass BCP-47+script codes directly.
        """
        if not text.strip():
            return ""

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True,
            max_length=512,
        ).to(self.device)

        forced_bos = self.tokenizer.lang_code_to_id[tgt_lang]
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=max_length,
                num_beams=4,
            )

        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    @staticmethod
    def resolve_lang(name_or_code: str) -> str:
        """Accept friendly name ('hindi') or raw code ('hin_Deva')."""
        return LANG_CODES.get(name_or_code.lower(), name_or_code)
