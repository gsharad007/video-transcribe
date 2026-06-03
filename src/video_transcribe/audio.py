"""Audio extraction/normalization via ffmpeg.

Whisper models expect 16 kHz mono 16-bit PCM. We let ffmpeg do the decoding so
any container/codec it understands (mp4, mkv, webm, mp3, m4a, ...) just works.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

WHISPER_SAMPLE_RATE = 16_000


class FFmpegNotFound(RuntimeError):
    """Raised when ffmpeg/ffprobe cannot be located on PATH."""


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise FFmpegNotFound(
            f"`{tool}` was not found on PATH. Install ffmpeg "
            "(https://ffmpeg.org/download.html) and make sure `ffmpeg` and "
            "`ffprobe` are runnable from your shell."
        )
    return path


def probe_duration(media: Path) -> float | None:
    """Return the media duration in seconds, or None if it can't be determined."""
    ffprobe = _require("ffprobe")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(media)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def extract_audio(media: Path, dest: Path) -> Path:
    """Decode `media` to a 16 kHz mono 16-bit PCM WAV at `dest`."""
    ffmpeg = _require("ffmpeg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [ffmpeg, "-y", "-i", str(media),
         "-vn",                              # drop any video stream
         "-ac", "1",                         # mono
         "-ar", str(WHISPER_SAMPLE_RATE),    # 16 kHz
         "-c:a", "pcm_s16le",                # 16-bit PCM
         str(dest)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract audio from {media}:\n{proc.stderr.strip()}"
        )
    return dest
