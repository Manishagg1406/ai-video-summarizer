from faster_whisper import WhisperModel
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

@dataclass
class Transcript:
    language: str
    full_text: str
    segments: List[TranscriptSegment]

class WhisperTranscriber:
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        compute = "float32"
        print(f"[Whisper] Loading '{model_size}' via faster-whisper ...")
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Transcript:
        segments_gen, info = self.model.transcribe(
            audio_path, language=language, beam_size=5
        )
        segments = [
            TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments_gen
        ]
        return Transcript(
            language=info.language,
            full_text=" ".join(s.text for s in segments),
            segments=segments,
        )
