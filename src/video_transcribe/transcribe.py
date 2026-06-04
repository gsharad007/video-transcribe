"""Thin wrapper around faster-whisper that yields plain dataclasses.

Keeping the faster-whisper import lazy (inside `transcribe`) means `--help`,
`--version` and the formatters/merge logic don't pay its heavy import cost, and
makes it easy to swap in a different backend (e.g. whisper.cpp + Vulkan) later
without touching the CLI, merge or formatters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# progress(segment, total_audio_seconds)
ProgressFn = Callable[["Segment", float], None]


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str  # includes faster-whisper's leading space, e.g. " hello"


@dataclass(frozen=True)
class Segment:
    index: int
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TranscriptionResult:
    language: str
    language_probability: float
    duration: float
    model: str
    segments: list[Segment]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()


def transcribe(
    audio_path: Path,
    *,
    model_size: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    vad_filter: bool = True,
    beam_size: int = 5,
    word_timestamps: bool = False,
    progress: ProgressFn | None = None,
) -> TranscriptionResult:
    """Transcribe a 16 kHz WAV and return all segments with timestamps."""
    from faster_whisper import WhisperModel  # heavy import, kept local

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=vad_filter,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
    )

    # faster-whisper returns `raw_segments` as a lazy generator; iterating it is
    # what actually drives decoding, so this loop is where the work happens.
    segments: list[Segment] = []
    for i, seg in enumerate(raw_segments):
        words: tuple[Word, ...] = ()
        if word_timestamps and seg.words:
            words = tuple(
                Word(start=w.start, end=w.end, text=w.word)
                for w in seg.words
                if w.start is not None and w.end is not None
            )
        s = Segment(index=i, start=seg.start, end=seg.end, text=seg.text, words=words)
        segments.append(s)
        if progress is not None:
            progress(s, info.duration)

    return TranscriptionResult(
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        model=model_size,
        segments=segments,
    )
