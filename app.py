"""
Streamlit frontend for VideoSummarizer.
Run: streamlit run ui/app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import streamlit as st
from PIL import Image

from pipeline import VideoSummarizer
from translator import LANG_CODES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VideoSummarizer AI",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 VideoSummarizer AI")
st.caption("ResNet50 · InceptionV3 · LSTM · Transformer · Whisper · DistilBART · NLLB-200")

# ── Sidebar: settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    whisper_size   = st.selectbox("Whisper model", ["tiny","base","small","medium"], index=1)
    temporal_kind  = st.radio("Temporal model", ["transformer", "lstm"])
    top_k          = st.slider("Keyframes to extract", 4, 16, 8)
    sample_fps     = st.slider("Sampling FPS", 0.5, 4.0, 1.0, step=0.5)
    target_lang    = st.selectbox(
        "Translate summary to",
        ["None"] + list(LANG_CODES.keys()),
    )
    save_hl        = st.checkbox("Generate highlight reel", value=False)

# ── Cache model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models (first run may take a while)...")
def load_model(whisper_size, temporal_kind):
    return VideoSummarizer(whisper_size=whisper_size, temporal_kind=temporal_kind)

vs = load_model(whisper_size, temporal_kind)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv","webm"])

if uploaded:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(uploaded.read())
        tmp_path = f.name

    st.video(tmp_path)

    if st.button("🚀 Summarize", type="primary"):
        with st.spinner("Processing video..."):
            lang = None if target_lang == "None" else target_lang
            hl_out = tempfile.mktemp(suffix=".mp4") if save_hl else None
            result = vs.summarize(
                tmp_path,
                top_k_frames=top_k,
                sample_fps=sample_fps,
                target_lang=lang,
                save_highlights=save_hl,
                highlight_out=hl_out,
            )

        # ── Results ───────────────────────────────────────────────────────
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📝 Summary")
            st.info(result.summary)
            if result.translated:
                st.subheader(f"🌍 Translated ({target_lang})")
                st.success(result.translated)

            st.subheader("📄 Full Transcript")
            with st.expander("Show transcript"):
                st.write(result.transcript)

            m = result.metadata
            st.markdown(f"""
            **Duration:** {result.duration:.1f}s &nbsp;|&nbsp;
            **Language:** `{result.language_detected}` &nbsp;|&nbsp;
            **Resolution:** {m['resolution'][0]}×{m['resolution'][1]}
            """)

        with col2:
            st.subheader(f"🖼️ Key Frames ({len(result.keyframes)})")
            n_cols = 4
            rows = [result.keyframes[i:i+n_cols]
                    for i in range(0, len(result.keyframes), n_cols)]
            for row_frames, row_times in zip(
                rows,
                [result.keyframe_timestamps[i:i+n_cols]
                 for i in range(0, len(result.keyframe_timestamps), n_cols)]
            ):
                cols = st.columns(len(row_frames))
                for c, frame, t in zip(cols, row_frames, row_times):
                    c.image(frame, caption=f"{t:.1f}s", width=300)

        if result.highlight_path and os.path.exists(result.highlight_path):
            st.subheader("🎞️ Highlight Reel")
            with open(result.highlight_path, "rb") as f:
                st.video(f.read())

        os.unlink(tmp_path)
