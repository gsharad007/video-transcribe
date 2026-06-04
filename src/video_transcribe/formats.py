"""Render a Conversation into human-readable txt / srt / vtt / json."""

from __future__ import annotations

import json
from dataclasses import dataclass

from video_transcribe.merge import Conversation
from video_transcribe.transcribe import TranscriptionResult


@dataclass(frozen=True)
class Meta:
    title: str
    language: str
    duration: float
    model: str
    diarized: bool

    @classmethod
    def from_result(cls, title: str, result: TranscriptionResult, diarized: bool) -> "Meta":
        return cls(title=title, language=result.language, duration=result.duration,
                   model=result.model, diarized=diarized)


def _clock(seconds: float) -> str:
    """H:MM:SS (drops the hour when zero) — for the readable transcript."""
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _stamp(seconds: float, sep: str) -> str:
    """HH:MM:SS<sep>mmm — for SRT (sep=',') and VTT (sep='.')."""
    millis = max(0, int(round(seconds * 1000)))
    h, millis = divmod(millis, 3_600_000)
    m, millis = divmod(millis, 60_000)
    s, millis = divmod(millis, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{millis:03d}"


def _header(meta: Meta, conv: Conversation) -> str:
    bits = [f"Language: {meta.language}", f"Duration: {_clock(meta.duration)}",
            f"Model: {meta.model}"]
    if meta.diarized:
        bits.append(f"Speakers: {len(conv.speakers)}")
    line = "  |  ".join(bits)
    rule = "=" * max(len(meta.title), len(line))
    return f"{meta.title}\n{line}\n{rule}"


def to_txt(conv: Conversation, meta: Meta) -> str:
    """Readable transcript: timestamped, speaker-labeled paragraphs."""
    blocks = [_header(meta, conv)]
    for u in conv.utterances:
        who = f"{u.speaker}: " if u.speaker else ""
        blocks.append(f"[{_clock(u.start)}] {who}{u.text}")
    return "\n\n".join(blocks) + "\n"


def _cue_text(speaker: str | None, text: str) -> str:
    return f"{speaker}: {text}" if speaker else text


def to_srt(conv: Conversation, meta: Meta) -> str:
    blocks = [
        f"{i}\n"
        f"{_stamp(s.start, ',')} --> {_stamp(s.end, ',')}\n"
        f"{_cue_text(s.speaker, s.text)}\n"
        for i, s in enumerate(conv.segments, start=1)
    ]
    return "\n".join(blocks)


def to_vtt(conv: Conversation, meta: Meta) -> str:
    blocks = ["WEBVTT\n"]
    blocks += [
        f"{_stamp(s.start, '.')} --> {_stamp(s.end, '.')}\n"
        f"{_cue_text(s.speaker, s.text)}\n"
        for s in conv.segments
    ]
    return "\n".join(blocks)


def to_json(conv: Conversation, meta: Meta) -> str:
    payload = {
        "title": meta.title,
        "language": meta.language,
        "duration": round(meta.duration, 3),
        "model": meta.model,
        "diarized": meta.diarized,
        "speakers": conv.speakers,
        "utterances": [
            {"start": round(u.start, 3), "end": round(u.end, 3),
             "speaker": u.speaker, "text": u.text}
            for u in conv.utterances
        ],
        "segments": [
            {"id": s.index, "start": round(s.start, 3), "end": round(s.end, 3),
             "speaker": s.speaker, "text": s.text}
            for s in conv.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


WRITERS = {
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
    "json": to_json,
}
