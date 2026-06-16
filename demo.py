"""
demo.py — quick end-to-end smoke test.
Usage:
    python demo.py --video path/to/video.mp4 --lang hindi
"""
import argparse
from pipeline import VideoSummarizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   required=True)
    parser.add_argument("--lang",    default=None,  help="e.g. hindi, french")
    parser.add_argument("--top_k",   type=int, default=8)
    parser.add_argument("--highlights", action="store_true")
    args = parser.parse_args()

    vs = VideoSummarizer()
    result = vs.summarize(
        args.video,
        top_k_frames=args.top_k,
        target_lang=args.lang,
        save_highlights=args.highlights,
        highlight_out="highlights.mp4" if args.highlights else None,
    )

    print("\n" + "="*60)
    print("LANGUAGE DETECTED :", result.language_detected)
    print("DURATION          :", f"{result.duration:.1f}s")
    print("\nTRANSCRIPT (first 300 chars):")
    print(result.transcript[:300], "...")
    print("\nSUMMARY:")
    print(result.summary)
    if result.translated:
        print("\nTRANSLATED:")
        print(result.translated)
    print(f"\nKEYFRAMES: {len(result.keyframes)} frames at "
          f"{[f'{t:.1f}s' for t in result.keyframe_timestamps]}")
    print(f"SEGMENTS : {result.highlight_segments}")
    if result.highlight_path:
        print(f"HIGHLIGHT: {result.highlight_path}")
    print("="*60)

    # Save keyframes
    for i, (img, t) in enumerate(
            zip(result.keyframes, result.keyframe_timestamps)):
        path = f"keyframe_{i:02d}_{t:.1f}s.jpg"
        img.save(path)
        print(f"Saved {path}")

if __name__ == "__main__":
    main()
