"""Render a TranscriptionResult into txt / srt / vtt / json."""

from __future__ import annotations

import json

from video_transcribe.transcribe import TranscriptionResult


def _timestamp(seconds: float, sep: str) -> str:
    """Format seconds as HH:MM:SS<sep>mmm (sep is ',' for SRT, '.' for VTT)."""
    millis = max(0, int(round(seconds * 1000)))
    h, millis = divmod(millis, 3_600_000)
    m, millis = divmod(millis, 60_000)
    s, millis = divmod(millis, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{millis:03d}"


def to_txt(result: TranscriptionResult) -> str:
    return "\n".join(s.text.strip() for s in result.segments) + "\n"


def to_srt(result: TranscriptionResult) -> str:
    blocks = [
        f"{s.index + 1}\n"
        f"{_timestamp(s.start, ',')} --> {_timestamp(s.end, ',')}\n"
        f"{s.text.strip()}\n"
        for s in result.segments
    ]
    return "\n".join(blocks)


def to_vtt(result: TranscriptionResult) -> str:
    blocks = ["WEBVTT\n"]
    blocks += [
        f"{_timestamp(s.start, '.')} --> {_timestamp(s.end, '.')}\n"
        f"{s.text.strip()}\n"
        for s in result.segments
    ]
    return "\n".join(blocks)


def to_json(result: TranscriptionResult) -> str:
    payload = {
        "language": result.language,
        "language_probability": round(result.language_probability, 4),
        "duration": round(result.duration, 3),
        "text": result.text,
        "segments": [
            {
                "id": s.index,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
            }
            for s in result.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


WRITERS = {
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
    "json": to_json,
}
