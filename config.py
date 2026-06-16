from dataclasses import dataclass, field
from typing import Literal
import torch

@dataclass
class Config:
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Visual
    visual_backbone: Literal["resnet50", "inceptionv3", "ensemble"] = "ensemble"
    frame_sample_fps: float = 1.0          # frames per second to sample
    top_k_frames: int = 8                  # keyframes to keep

    # Temporal
    temporal_model: Literal["lstm", "transformer"] = "transformer"
    lstm_hidden: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 4

    # Audio
    whisper_model: Literal["tiny","base","small","medium","large"] = "base"
    whisper_language: str = None           # None = auto-detect

    # Summarization
    summarizer_model: str = "sshleifer/distilbart-cnn-12-6"
    max_summary_tokens: int = 256
    min_summary_tokens: int = 56

    # Translation
    translation_model: str = "facebook/nllb-200-distilled-600M"
    target_language: str = "eng_Latn"     # NLLB language code

    # Paths
    output_dir: str = "outputs"
    cache_dir: str = ".cache"

CONFIG = Config()
