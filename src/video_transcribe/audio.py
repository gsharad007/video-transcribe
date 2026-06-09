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


def extract_audio(media: Path, dest: Path, *, stream_index: int | None = None) -> Path:
    """Decode `media` to a 16 kHz mono 16-bit PCM WAV at `dest`.

    `stream_index` (0-based) selects the Nth audio stream for multi-track inputs,
    e.g. a ReLive recording with a separate microphone track.
    """
    ffmpeg = _require("ffmpeg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(media)]
    if stream_index is not None:
        cmd += ["-map", f"0:a:{stream_index}"]    # select one audio track
    cmd += ["-vn",                                # drop any video stream
            "-ac", "1",                           # mono
            "-ar", str(WHISPER_SAMPLE_RATE),      # 16 kHz
            "-c:a", "pcm_s16le",                  # 16-bit PCM
            str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract audio from {media}:\n{proc.stderr.strip()}"
        )
    return dest


def probe_streams(media: Path) -> list[dict]:
    """List audio streams: a_index (for -map 0:a:N), codec, channels, title, language."""
    ffprobe = _require("ffprobe")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a",
         "-show_entries",
         "stream=index,codec_name,channels,sample_rate:stream_tags=title,language",
         "-of", "json", str(media)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return []
    try:
        streams = json.loads(proc.stdout or "{}").get("streams", [])
    except json.JSONDecodeError:
        return []
    out: list[dict] = []
    for a_index, s in enumerate(streams):
        tags = s.get("tags") or {}
        out.append({
            "a_index": a_index,
            "codec": s.get("codec_name"),
            "channels": s.get("channels"),
            "sample_rate": s.get("sample_rate"),
            "title": tags.get("title"),
            "language": tags.get("language"),
        })
    return out
