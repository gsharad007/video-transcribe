"""Combine ASR segments with diarization turns into clean, readable utterances.

The pipeline here is independent of faster-whisper / pyannote (it works on the
plain dataclasses), so it's cheap to import and easy to unit-test with synthetic
speaker turns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from video_transcribe.diarize import SpeakerTurn
from video_transcribe.transcribe import Segment

# Phrases Whisper habitually hallucinates over silence/music/applause. Compared
# case-insensitively after stripping punctuation; only *whole-segment* matches
# are dropped, so real sentences containing these words are untouched.
_HALLUCINATIONS = {
    "you", "thank you", "thank you.", "thanks for watching",
    "thank you for watching", "please subscribe", "subscribe",
    "bye", "bye bye", "subtitles by the amara.org community", "music",
    "applause", "silence",
}


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
    """Speaker whose turn overlaps [start, end] most; nearest turn if none overlap."""
    best_spk, best = None, 0.0
    for t in turns:
        ov = _overlap(start, end, t.start, t.end)
        if ov > best:
            best, best_spk = ov, t.speaker
    if best_spk is not None:
        return best_spk
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


def _smooth(words: list[list]) -> None:
    """Flip a lone word's speaker when both neighbours agree (de-noises timings)."""
    for i in range(1, len(words) - 1):
        prev, cur, nxt = words[i - 1][3], words[i][3], words[i + 1][3]
        if cur != prev and prev == nxt and prev is not None:
            words[i][3] = prev


def _group_by_speaker(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Utterance]:
    """Word-level speaker assignment, then merge consecutive same-speaker words."""
    words: list[list] = []  # [start, end, text, speaker]
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append([w.start, w.end, w.text, _speaker_at(w.start, w.end, turns)])
        else:
            words.append([seg.start, seg.end, seg.text, _speaker_at(seg.start, seg.end, turns)])
    if not words:
        return []
    _smooth(words)

    utterances: list[Utterance] = []
    cs, ce, ct, cspk = words[0][0], words[0][1], [words[0][2]], words[0][3]
    for start, end, text, spk in words[1:]:
        if spk == cspk:
            ct.append(text)
            ce = end
        else:
            utterances.append(Utterance(cs, ce, cspk, "".join(ct).strip()))
            cs, ce, ct, cspk = start, end, [text], spk
    utterances.append(Utterance(cs, ce, cspk, "".join(ct).strip()))
    return [u for u in utterances if u.text]


def _group_by_pause(
    segments: list[Segment], max_gap: float = 1.5, max_chars: int = 320
) -> list[Utterance]:
    """No-diarization fallback: paragraph breaks on pauses / length."""
    utterances: list[Utterance] = []
    cs = ce = None
    buf: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        gap = 0.0 if cs is None else seg.start - ce
        joined = " ".join(buf)
        if cs is None:
            cs = seg.start
        elif gap > max_gap or len(joined) >= max_chars:
            utterances.append(Utterance(cs, ce, None, joined))
            cs, buf = seg.start, []
        buf.append(text)
        ce = seg.end
    if buf:
        utterances.append(Utterance(cs, ce, None, " ".join(buf)))
    return utterances


def _tidy(text: str) -> str:
    """Light, non-destructive readability pass: spacing, capital, terminal stop."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?…\"')]":
        text += "."
    return text


def _friendly_names(diarized: list[DiarizedSegment]) -> dict[str, str]:
    """Map raw pyannote labels to 'Speaker 1', 'Speaker 2', ... by first appearance."""
    mapping: dict[str, str] = {}
    for d in diarized:
        if d.speaker is not None and d.speaker not in mapping:
            mapping[d.speaker] = f"Speaker {len(mapping) + 1}"
    return mapping


def build_conversation(
    segments: list[Segment],
    turns: list[SpeakerTurn],
    *,
    tidy: bool = True,
) -> Conversation:
    """Clean ASR output and fuse it with diarization into a Conversation."""
    segments = clean_segments(segments)

    if turns:
        raw_utterances = _group_by_speaker(segments, turns)
        diarized = [
            DiarizedSegment(s.index, s.start, s.end, s.text.strip(),
                            _majority_speaker(s, turns))
            for s in segments
        ]
        names = _friendly_names(diarized)
        diarized = [replace(d, speaker=names.get(d.speaker)) for d in diarized]
        utterances = [replace(u, speaker=names.get(u.speaker)) for u in raw_utterances]
        speakers = list(dict.fromkeys(names.values()))
    else:
        utterances = _group_by_pause(segments)
        diarized = [
            DiarizedSegment(s.index, s.start, s.end, s.text.strip(), None)
            for s in segments
        ]
        speakers = []

    if tidy:
        utterances = [replace(u, text=_tidy(u.text)) for u in utterances]

    return Conversation(segments=diarized, utterances=utterances, speakers=speakers)
