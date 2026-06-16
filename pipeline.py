"""
VideoSummarizer: main pipeline orchestrating all models.

Usage:
    from pipeline import VideoSummarizer
    vs = VideoSummarizer()
    result = vs.summarize("lecture.mp4", target_lang="hindi")
"""
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from PIL import Image

from config import CONFIG
from visual_encoder  import EnsembleVisualEncoder, encode_frames
from temporal_model  import build_temporal_model
from audio_pipeline  import WhisperTranscriber, Transcript
from summarizer      import DistilBARTSummarizer
from translator      import NLLB200Translator
from video_processor  import (extract_frames, extract_audio,
                                     get_video_metadata, create_highlight_reel)
from frame_selector   import select_keyframes_mmr, scores_to_segments


@dataclass
class SummaryResult:
    # Text
    transcript:     str
    summary:        str
    translated:     Optional[str]
    language_detected: str

    # Visual
    keyframe_indices:   List[int]
    keyframe_timestamps: List[float]
    keyframes:          List[Image.Image]

    # Video segments
    highlight_segments: List[Tuple[float, float]]
    highlight_path:     Optional[str]

    # Meta
    duration:    float
    metadata:    dict = field(default_factory=dict)


class VideoSummarizer:
    def __init__(self,
                 device: Optional[str] = None,
                 whisper_size: str = CONFIG.whisper_model,
                 temporal_kind: str = CONFIG.temporal_model):
        self.device = device or CONFIG.device
        print(f"[VideoSummarizer] Using device: {self.device}")

        # Visual
        self.visual_enc = EnsembleVisualEncoder(hidden_dim=512).to(self.device)
        self.visual_enc.eval()

        # Temporal
        self.temporal = build_temporal_model(
            temporal_kind, input_dim=512,
        ).to(self.device)
        self.temporal.eval()

        # Audio & NLP
        self.transcriber  = WhisperTranscriber(whisper_size, self.device)
        self.summarizer   = DistilBARTSummarizer(self.device)
        self.translator   = NLLB200Translator(self.device)

    # ─────────────────────────────────────────────────────────────────────────

    def summarize(self,
                  video_path: str,
                  top_k_frames: int = CONFIG.top_k_frames,
                  sample_fps: float = CONFIG.frame_sample_fps,
                  target_lang: Optional[str] = None,
                  save_highlights: bool = False,
                  highlight_out: Optional[str] = None) -> SummaryResult:

        meta = get_video_metadata(video_path)
        print(f"[Pipeline] Duration={meta['duration']:.1f}s  "
              f"FPS={meta['fps']:.1f}  Res={meta['resolution']}")

        # ── 1. Extract frames ─────────────────────────────────────────────
        print("[Pipeline] Extracting frames ...")
        frames, timestamps = extract_frames(video_path, sample_fps=sample_fps)
        print(f"[Pipeline] {len(frames)} frames sampled.")

        # ── 2. Visual embeddings ──────────────────────────────────────────
        print("[Pipeline] Encoding frames (ResNet50 + InceptionV3) ...")
        embeddings = encode_frames(frames, self.visual_enc,
                                   self.device)          # (T, 512)

        # ── 3. Temporal scoring ───────────────────────────────────────────
        print(f"[Pipeline] Scoring with temporal model ...")
        with torch.no_grad():
            x = embeddings.unsqueeze(0).to(self.device)  # (1, T, 512)
            scores = self.temporal(x).cpu()               # (T,)

        # ── 4. Keyframe selection (MMR) ───────────────────────────────────
        keyframe_idx = select_keyframes_mmr(scores, embeddings, top_k=top_k_frames)
        keyframes    = [frames[i]     for i in keyframe_idx]
        kf_times     = [timestamps[i] for i in keyframe_idx]

        # ── 5. Highlight segments ─────────────────────────────────────────
        segments = scores_to_segments(scores, timestamps, threshold=0.5)

        # ── 6. Audio → Transcript ─────────────────────────────────────────
        print("[Pipeline] Extracting audio & transcribing with Whisper ...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
        try:
            extract_audio(video_path, audio_path)
            transcript_obj: Transcript = self.transcriber.transcribe(audio_path)
        finally:
            os.unlink(audio_path)

        print(f"[Pipeline] Language detected: {transcript_obj.language}")
        transcript_text = transcript_obj.full_text

        # ── 7. Summarize transcript ───────────────────────────────────────
        print("[Pipeline] Summarizing with DistilBART ...")
        summary = self.summarizer.summarize(transcript_text)

        # ── 8. Translation (optional) ─────────────────────────────────────
        translated = None
        if target_lang:
            tgt_code = NLLB200Translator.resolve_lang(target_lang)
            print(f"[Pipeline] Translating to {tgt_code} with NLLB-200 ...")
            translated = self.translator.translate(
                summary, tgt_lang=tgt_code
            )

        # ── 9. Highlight reel (optional) ──────────────────────────────────
        hl_path = None
        if save_highlights and segments:
            hl_path = highlight_out or "highlights.mp4"
            print(f"[Pipeline] Creating highlight reel → {hl_path}")
            create_highlight_reel(video_path, segments, hl_path)

        return SummaryResult(
            transcript          = transcript_text,
            summary             = summary,
            translated          = translated,
            language_detected   = transcript_obj.language,
            keyframe_indices    = keyframe_idx,
            keyframe_timestamps = kf_times,
            keyframes           = keyframes,
            highlight_segments  = segments,
            highlight_path      = hl_path,
            duration            = meta["duration"],
            metadata            = meta,
        )
