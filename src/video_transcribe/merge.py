"""Combine ASR segments with diarization turns into clean, readable utterances.

Speakers are assigned per ASR *segment* (majority overlap of the segment's words
with the diarization turns), which avoids the mid-phrase speaker flips that
per-word assignment can produce.

Punctuation is restored over the *full concatenated utterance stream* (not per
segment), so the model finds real sentence boundaries instead of inserting a
false break at every Whisper segment edge. Subtitle cues (`segments`) are
punctuated per-segment, which is fine for short on-screen fragments.

This module works on the plain dataclasses (no faster-whisper / pyannote /
punctuators import), so it's cheap to import and easy to unit-test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from video_transcribe.diarize import SpeakerTurn
from video_transcribe.transcribe import Segment

# Phrases Whisper habitually hallucinates over silence/music/applause. Compared
# case-insensitively after stripping punctuation; only *whole-segment* matches
# are dropped, so real sentences containing these words are untouched.
_HALLUCINATIONS = {
    "you", "thank you", "thanks for watching", "thank you for watching",
    "please subscribe", "subscribe", "bye", "bye bye", "okay", "so",
    "subtitles by the amara.org community", "music", "applause", "silence",
}

# A callable that re-punctuates a batch of texts (1:1 in/out). See punctuate.py.
Punctuator = Callable[[list[str]], list[str]]


@dataclass(frozen=True)
class DiarizedSegment:
    """A Whisper segment annotated with its (majority) speaker."""
    index: int
    start: float
    end: float
    text: str
    speaker: str | None


@dataclass(frozen=True)
class Utterance:
    """A readable, speaker-contiguous block of speech."""
    start: float
    end: float
    speaker: str | None
    text: str


@dataclass(frozen=True)
class Conversation:
    segments: list[DiarizedSegment]   # subtitle-granularity (for srt/vtt)
    utterances: list[Utterance]       # speaker/paragraph-granularity (for txt)
    speakers: list[str]               # friendly names, first-appearance order


def _norm(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE).strip().lower()


def clean_segments(segments: list[Segment]) -> list[Segment]:
    """Drop empty / hallucination-only segments and collapse stuck repeats."""
    out: list[Segment] = []
    prev_norm: str | None = None
    for seg in segments:
        norm = _norm(seg.text)
        if not norm or norm in _HALLUCINATIONS or norm == prev_norm:
            continue
        out.append(seg)
        prev_norm = norm
    return out


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _speaker_at(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    """Nearest speaker turn to [start, end] (fallback when there's no overlap)."""
    if not turns:
        return None
    mid = (start + end) / 2
    return min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))).speaker


def _majority_speaker(seg: Segment, turns: list[SpeakerTurn]) -> str | None:
    """Speaker with the most overlap across all of a segment's words."""
    spans = [(w.start, w.end) for w in seg.words] or [(seg.start, seg.end)]
    by_spk: dict[str, float] = {}
    for s, e in spans:
        for t in turns:
            ov = _overlap(s, e, t.start, t.end)
            if ov > 0:
                by_spk[t.speaker] = by_spk.get(t.speaker, 0.0) + ov
    if by_spk:
        return max(by_spk, key=by_spk.get)
    return _speaker_at(seg.start, seg.end, turns)


def _tidy(text: str) -> str:
    """Light fallback when no punctuation model: spacing, capital, terminal stop."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?…\"')]":
        text += "."
    return text


def _friendly_names(raw_speakers: list[str | None]) -> dict[str, str]:
    """Map raw pyannote labels to 'Speaker 1', 'Speaker 2', ... by first appearance."""
    mapping: dict[str, str] = {}
    for spk in raw_speakers:
        if spk is not None and spk not in mapping:
            mapping[spk] = f"Speaker {len(mapping) + 1}"
    return mapping


def _group_indices(
    segments: list[Segment], seg_speaker: list[str | None], *,
    diarized: bool, max_gap: float = 1.5, max_chars: int = 700,
) -> list[dict]:
    """Group consecutive segment indices into utterances.

    Diarized: break on speaker change. Otherwise: break on a pause or length.
    """
    groups: list[dict] = []
    cur: dict | None = None
    for i, s in enumerate(segments):
        if cur is None:
            cur = {"start": s.start, "end": s.end, "speaker": seg_speaker[i],
                   "idxs": [i], "chars": len(s.text)}
            continue
        if diarized:
            split = seg_speaker[i] != cur["speaker"]
        else:
            split = (s.start - cur["end"]) > max_gap or cur["chars"] >= max_chars
        if split:
            groups.append(cur)
            cur = {"start": s.start, "end": s.end, "speaker": seg_speaker[i],
                   "idxs": [i], "chars": len(s.text)}
        else:
            cur["idxs"].append(i)
            cur["end"] = s.end
            cur["chars"] += len(s.text)
    if cur is not None:
        groups.append(cur)
    return groups


def build_conversation(
    segments: list[Segment],
    turns: list[SpeakerTurn],
    *,
    tidy: bool = True,
    punctuator: Punctuator | None = None,
) -> Conversation:
    """Clean ASR output, assign speakers, restore punctuation, and group."""
    segments = clean_segments(segments)

    # 1. speaker per segment + friendly relabeling
    if turns:
        raw_spk = [_majority_speaker(s, turns) for s in segments]
        names = _friendly_names(raw_spk)
        seg_spk = [names.get(x) for x in raw_spk]
        speakers = list(dict.fromkeys(names.values()))
    else:
        seg_spk = [None] * len(segments)
        speakers = []

    raw = [s.text.strip() for s in segments]

    # 2. group consecutive segments into utterances
    groups = _group_indices(segments, seg_spk, diarized=bool(turns))
    utt_raw = [" ".join(raw[i] for i in g["idxs"]) for g in groups]

    # 3. readability: full-stream punctuation for utterances (correct sentence
    #    boundaries) and per-segment punctuation for subtitle cues; else tidy.
    if punctuator is not None:
        utt_text = punctuator(utt_raw)
        seg_text = punctuator(raw)
    elif tidy:
        utt_text = [_tidy(t) for t in utt_raw]
        seg_text = [_tidy(t) for t in raw]
    else:
        utt_text = utt_raw
        seg_text = raw

    diarized = [
        DiarizedSegment(s.index, s.start, s.end, (seg_text[i].strip() or raw[i]), seg_spk[i])
        for i, s in enumerate(segments)
    ]
    utterances = [
        Utterance(g["start"], g["end"], g["speaker"], (utt_text[k].strip() or utt_raw[k]))
        for k, g in enumerate(groups)
    ]
    return Conversation(segments=diarized, utterances=utterances, speakers=speakers)
