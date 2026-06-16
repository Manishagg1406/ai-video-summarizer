from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class DistilBARTSummarizer:
    MODEL_ID = "sshleifer/distilbart-cnn-12-6"
    MAX_INPUT_TOKENS = 1024

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DistilBART] Loading on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()

    def summarize(self, text: str, max_length: int = 256, min_length: int = 56) -> str:
        if not text.strip():
            return ""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.MAX_INPUT_TOKENS
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_length=max_length,
                min_length=min_length, num_beams=4
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
