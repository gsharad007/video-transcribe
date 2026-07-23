"""Standalone: merge a video + separate mic file into one MKV.

Output has three audio tracks: a default **Mix** (desktop+mic), plus separate
**Desktop** and **Mic**. Press play -> hear both; switch tracks to isolate.

Usage:
  uv run python -m video_transcribe.mux VIDEO.mp4 MIC.m4a [-o OUT.mkv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from video_transcribe import audio


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="video-transcribe-mux",
        description="Combine a video (with its desktop audio) + a separate mic file "
                    "into one MKV: default Mix track + separate Desktop and Mic tracks.",
    )
    p.add_argument("video", type=Path, help="video file (with desktop/system audio)")
    p.add_argument("mic", type=Path, help="separate microphone audio file")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="output .mkv (default: <video>.mkv, or <video>.with-mic.mkv "
                        "when the source is itself an .mkv)")
    args = p.parse_args(argv)

    out = args.output or audio.muxed_output_path(args.video)
    try:
        audio.mux_tracks(args.video, args.mic, out)
    except (audio.FFmpegNotFound, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
