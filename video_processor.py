"""
Video Processor: FFmpeg + OpenCV + MoviePy utilities.
- Frame extraction at target FPS
- Audio extraction to WAV
- Highlight reel creation
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
from typing import Optional
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips


# ── Frame Extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str,
                   sample_fps: float = 1.0) -> Tuple[List[Image.Image], List[float]]:
    """
    Extract frames at `sample_fps` using OpenCV.
    Returns: (frames as PIL Images, timestamps in seconds)
    """
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(native_fps / sample_fps)))

    frames, timestamps = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            timestamps.append(idx / native_fps)
        idx += 1

    cap.release()
    return frames, timestamps


def get_video_metadata(video_path: str) -> dict:
    """Return duration, fps, resolution via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "duration":   n_frames / fps,
        "fps":        fps,
        "n_frames":   int(n_frames),
        "resolution": (width, height),
    }


# ── Audio Extraction ──────────────────────────────────────────────────────────

def extract_audio(video_path: str,
                  out_path: Optional[str] = None) -> str:
    """
    Extract audio track to WAV using FFmpeg.
    Returns path to the WAV file.
    """
    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = tmp.name

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",  # PCM WAV
        "-ar", "16000",           # 16 kHz (Whisper native)
        "-ac", "1",               # mono
        out_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True)
    return out_path


# ── Highlight Reel ────────────────────────────────────────────────────────────

def create_highlight_reel(video_path: str,
                          segments: List[Tuple[float, float]],
                          out_path: str,
                          padding: float = 0.5) -> str:
    """
    Create a highlight reel by concatenating the given time segments.
    segments: list of (start_sec, end_sec)
    padding : seconds added around each segment
    """
    clip = VideoFileClip(video_path)
    duration = clip.duration
    sub_clips = []

    for start, end in segments:
        t0 = max(0.0, start - padding)
        t1 = min(duration, end + padding)
        sub_clips.append(clip.subclip(t0, t1))

    if not sub_clips:
        clip.close()
        raise ValueError("No valid segments for highlight reel.")

    final = concatenate_videoclips(sub_clips, method="compose")
    final.write_videofile(out_path, codec="libx264",
                          audio_codec="aac", logger=None)
    clip.close()
    return out_path


# ── optional import fix ───────────────────────────────────────────────────────

