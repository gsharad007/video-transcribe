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
from video_transcribe.transcribe import Segment, TranscriptionResult

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


def _friendly_names(
    raw_speakers: list[str | None], known: dict[str, str] | None = None,
) -> dict[str, str]:
    """Map raw pyannote labels to names.

    A label present in `known` (e.g. a confident voiceprint match) gets that
    real name; every other label gets 'Speaker 1', 'Speaker 2', ... numbered
    by first appearance among the *unmatched* labels only.
    """
    known = known or {}
    mapping: dict[str, str] = {}
    n = 0
    for spk in raw_speakers:
        if spk is None or spk in mapping:
            continue
        if spk in known:
            mapping[spk] = known[spk]
        else:
            n += 1
            mapping[spk] = f"Speaker {n}"
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


def _assemble(
    segments: list[Segment],
    seg_spk: list[str | None],
    speakers: list[str],
    *,
    tidy: bool,
    punctuator: Punctuator | None,
) -> Conversation:
    """Given segments (in display order) + a final per-segment speaker, build it."""
    has_speakers = any(s is not None for s in seg_spk)
    raw = [s.text.strip() for s in segments]

    groups = _group_indices(segments, seg_spk, diarized=has_speakers)
    utt_raw = [" ".join(raw[i] for i in g["idxs"]) for g in groups]

    # Full-stream punctuation for utterances (correct sentence boundaries) and
    # per-segment punctuation for subtitle cues; else a light tidy.
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
        DiarizedSegment(i, s.start, s.end, (seg_text[i].strip() or raw[i]), seg_spk[i])
        for i, s in enumerate(segments)
    ]
    utterances = [
        Utterance(g["start"], g["end"], g["speaker"], (utt_text[k].strip() or utt_raw[k]))
        for k, g in enumerate(groups)
    ]
    return Conversation(segments=diarized, utterances=utterances, speakers=speakers)


def build_conversation(
    segments: list[Segment],
    turns: list[SpeakerTurn],
    *,
    tidy: bool = True,
    punctuator: Punctuator | None = None,
    voice_names: dict[str, str] | None = None,
) -> Conversation:
    """Clean ASR output, assign speakers from diarization turns, and assemble.

    `voice_names` maps raw pyannote labels to real names (e.g. from a confident
    voiceprint match, see voiceprint.py) -- labels not in it fall back to the
    generic 'Speaker N'.
    """
    segments = clean_segments(segments)
    if turns:
        raw_spk = [_majority_speaker(s, turns) for s in segments]
        names = _friendly_names(raw_spk, voice_names)
        seg_spk = [names.get(x) for x in raw_spk]
        speakers = list(dict.fromkeys(names.values()))
    else:
        seg_spk = [None] * len(segments)
        speakers = []
    return _assemble(segments, seg_spk, speakers, tidy=tidy, punctuator=punctuator)


def build_conversation_from_tracks(
    track_results: list[tuple[str, TranscriptionResult]],
    *,
    tidy: bool = True,
    punctuator: Punctuator | None = None,
) -> Conversation:
    """Merge per-track transcripts (each a fixed speaker) into one timeline.

    `track_results` is a list of (speaker_label, TranscriptionResult). Each
    track's segments are cleaned, tagged with that speaker, then all segments are
    interleaved by start time -- no acoustic diarization needed, because the
    speaker is known from which track the audio came from (e.g. mic vs desktop).
    """
    tagged_tracks = [
        [(seg, speaker) for seg in clean_segments(result.segments)]
        for speaker, result in track_results
    ]
    return build_conversation_from_tagged(tagged_tracks, tidy=tidy, punctuator=punctuator)


def diarized_track(
    result: TranscriptionResult, turns: list[SpeakerTurn],
    *, voice_names: dict[str, str] | None = None,
) -> list[tuple[Segment, str | None]]:
    """Clean a track's segments and tag each with its acoustically diarized speaker.

    Used for a track that itself mixes multiple people (e.g. a group meeting's
    desktop/system audio), as opposed to a track that is already a single known
    speaker. Speakers come out as friendly labels ("Speaker 1", "Speaker 2", ...)
    by order of first appearance -- rename them afterwards (e.g. via correct.py) --
    unless `voice_names` (a confident voiceprint match, see voiceprint.py) already
    resolves a label to a real name.
    """
    segments = clean_segments(result.segments)
    if not turns:
        return [(s, None) for s in segments]
    raw_spk = [_majority_speaker(s, turns) for s in segments]
    names = _friendly_names(raw_spk, voice_names)
    return [(s, names.get(r)) for s, r in zip(segments, raw_spk)]


def build_conversation_from_tagged(
    tagged_tracks: list[list[tuple[Segment, str | None]]],
    *,
    tidy: bool = True,
    punctuator: Punctuator | None = None,
) -> Conversation:
    """Merge several already speaker-tagged tracks into one timeline.

    Lets a diarized track (`diarized_track`, multiple speakers) and fixed-speaker
    tracks (one name per track, e.g. a separate mic) be combined into a single
    conversation ordered by start time.
    """
    tagged = [pair for track in tagged_tracks for pair in track]
    tagged.sort(key=lambda t: (t[0].start, t[0].end))
    segments = [seg for seg, _ in tagged]
    seg_spk = [spk for _, spk in tagged]
    speakers = list(dict.fromkeys(spk for spk in seg_spk if spk is not None))
    return _assemble(segments, seg_spk, speakers, tidy=tidy, punctuator=punctuator)
