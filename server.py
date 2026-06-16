"""
Flask REST API for VideoSummarizer.
Run: python api/server.py

Endpoints:
  POST /summarize      — upload video, get JSON summary
  GET  /health         — liveness check
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import base64
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from pipeline import VideoSummarizer
from config import CONFIG

app = Flask(__name__)
CORS(app)

# Lazy singleton
_summarizer = None

def get_summarizer() -> VideoSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = VideoSummarizer()
    return _summarizer


def pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "device": CONFIG.device})


@app.post("/summarize")
def summarize():
    """
    Multipart form fields:
      video        — required, video file
      top_k        — int, default 8
      sample_fps   — float, default 1.0
      target_lang  — string, e.g. 'hindi', optional
      highlights   — 'true'/'false', default false
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    f          = request.files["video"]
    top_k      = int(request.form.get("top_k", 8))
    sample_fps = float(request.form.get("sample_fps", 1.0))
    target_lang = request.form.get("target_lang") or None
    save_hl    = request.form.get("highlights", "false").lower() == "true"

    suffix = os.path.splitext(f.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
        video_path = tmp.name

    try:
        vs = get_summarizer()
        hl_out = tempfile.mktemp(suffix=".mp4") if save_hl else None
        result = vs.summarize(
            video_path,
            top_k_frames=top_k,
            sample_fps=sample_fps,
            target_lang=target_lang,
            save_highlights=save_hl,
            highlight_out=hl_out,
        )

        # Encode keyframes as base64 JPEG
        kf_b64 = [pil_to_b64(img) for img in result.keyframes]

        payload = {
            "transcript":        result.transcript,
            "summary":           result.summary,
            "translated":        result.translated,
            "language_detected": result.language_detected,
            "duration":          result.duration,
            "keyframe_timestamps": result.keyframe_timestamps,
            "keyframes_b64":     kf_b64,
            "highlight_segments": result.highlight_segments,
        }

        if save_hl and result.highlight_path and \
                os.path.exists(result.highlight_path):
            with open(result.highlight_path, "rb") as hf:
                payload["highlight_video_b64"] = \
                    base64.b64encode(hf.read()).decode()
            os.unlink(result.highlight_path)

        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(video_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
