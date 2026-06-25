---
title: Summize
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---


# 🎬 VideoSummarizer AI

A production-grade video summarization system combining computer vision, speech recognition, and multilingual NLP.

## Architecture

```
Video Input
    ├── 🎞️  Visual Pipeline    → FFmpeg → OpenCV → ResNet50 / InceptionV3 → LSTM / Transformer
    ├── 🎙️  Audio Pipeline     → FFmpeg → OpenAI Whisper → DistilBART
    └── 🌍  Translation        → Facebook NLLB-200 (200 languages)
              ↓
         Fusion Layer (text + visual embeddings)
              ↓
    📝 Summary + Key Frames + Timeline
              ↓
    🖥️  Streamlit UI  /  🔌  Flask REST API
```

## Tech Stack

| Category | Tools |
|----------|-------|
| Visual DL | ResNet50, InceptionV3, LSTM, Transformers |
| Audio NLP | OpenAI Whisper, DistilBART |
| Translation | Facebook NLLB-200 |
| Video | FFmpeg, OpenCV, MoviePy |
| Frameworks | PyTorch, Streamlit, Flask |

## Installation

```bash
# 1. Clone and enter
git clone https://github.com/youruser/video-summarizer
cd video-summarizer

# 2. Create environment
conda create -n videosumm python=3.10
conda activate videosumm

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install FFmpeg (system-level)
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg

# 5. Download models (first run auto-downloads)
python scripts/download_models.py
```

## Usage

### Streamlit UI
```bash
streamlit run ui/app.py
```

### Flask API
```bash
python api/server.py
# POST /summarize  with multipart video file
```

### Python SDK
```python
from video_summarizer import VideoSummarizer

vs = VideoSummarizer(device="cuda")
result = vs.summarize("myvideo.mp4", lang="en", top_k_frames=8)
print(result.summary)
print(result.transcript)
result.save_highlights("highlights.mp4")
```

## Project Structure

```
video_summarizer/
├── models/
│   ├── visual_encoder.py      # ResNet50 + InceptionV3 ensemble
│   ├── temporal_model.py      # LSTM + Transformer for sequence modeling
│   ├── audio_pipeline.py      # Whisper transcription
│   ├── summarizer.py          # DistilBART summarization
│   ├── translator.py          # NLLB-200 translation
│   └── fusion.py              # Multimodal fusion layer
├── utils/
│   ├── video_processor.py     # FFmpeg + OpenCV + MoviePy
│   ├── frame_selector.py      # Keyframe extraction algorithms
│   └── helpers.py             # Utilities
├── api/
│   ├── server.py              # Flask REST API
│   └── routes.py              # API endpoints
├── ui/
│   └── app.py                 # Streamlit frontend
├── pipeline.py                # Main orchestrator
├── requirements.txt
└── config.py
```
