"""Deterministic, model-free checks for the merge + format logic.

Run: uv run python tests/smoke.py
"""

from __future__ import annotations

from video_transcribe import formats, merge
from video_transcribe.diarize import SpeakerTurn
from video_transcribe.transcribe import Segment, TranscriptionResult, Word


def _w(t: str, s: float, e: float) -> Word:
    return Word(start=s, end=e, text=t)


def make_segments() -> list[Segment]:
    return [
        Segment(0, 0.0, 3.0, " Hello everyone, welcome.",
                (_w(" Hello", 0.1, 0.5), _w(" everyone,", 0.6, 1.2), _w(" welcome.", 1.3, 2.0))),
        Segment(1, 3.0, 6.0, " thanks for having me",
                (_w(" thanks", 3.1, 3.5), _w(" for", 3.6, 3.8),
                 _w(" having", 3.9, 4.3), _w(" me", 4.4, 4.6))),
        # pure hallucination over trailing silence -> must be dropped
        Segment(2, 6.2, 6.5, " you", (_w(" you", 6.2, 6.5),)),
    ]


TURNS = [
    SpeakerTurn(0.0, 3.0, "SPEAKER_01"),   # note: raw labels are out of order on purpose
    SpeakerTurn(3.0, 6.0, "SPEAKER_00"),
]

RESULT = TranscriptionResult("en", 0.99, 6.5, "test-model", make_segments())


def test_diarized():
    conv = merge.build_conversation(make_segments(), TURNS, tidy=True)

    assert conv.speakers == ["Speaker 1", "Speaker 2"], conv.speakers
    assert len(conv.utterances) == 2, conv.utterances
    u0, u1 = conv.utterances
    assert u0.speaker == "Speaker 1" and u0.text == "Hello everyone, welcome.", u0
    assert u1.speaker == "Speaker 2" and u1.text == "Thanks for having me.", u1
    # hallucination dropped everywhere
    assert all("you" != u.text.lower().strip(".") for u in conv.utterances)
    assert len(conv.segments) == 2, "hallucination segment should be gone"
    assert conv.segments[0].speaker == "Speaker 1"
    assert conv.segments[1].speaker == "Speaker 2"
    print("[ok] diarized merge: speakers + grouping + cleaning + tidy")


def test_no_diarize():
    conv = merge.build_conversation(make_segments(), [], tidy=True)
    assert conv.speakers == []
    assert all(u.speaker is None for u in conv.utterances)
    assert len(conv.segments) == 2
    joined = " ".join(u.text for u in conv.utterances)
    assert "you" not in joined.lower().split()
    print("[ok] no-diarize merge: pause grouping + cleaning")


def test_formats():
    conv = merge.build_conversation(make_segments(), TURNS, tidy=True)
    meta = formats.Meta.from_result("clip.mp4", RESULT, diarized=True)

    txt = formats.to_txt(conv, meta)
    assert "Speaker 1: Hello everyone, welcome." in txt
    assert "Speakers: 2" in txt and "clip.mp4" in txt

    srt = formats.to_srt(conv, meta)
    assert "1\n00:00:00,000 --> 00:00:03,000\nSpeaker 1:" in srt

    vtt = formats.to_vtt(conv, meta)
    assert vtt.startswith("WEBVTT")

    js = formats.to_json(conv, meta)
    assert '"speakers"' in js and '"Speaker 1"' in js
    print("[ok] formats: txt / srt / vtt / json render with speakers")
    print("\n----- sample txt -----")
    print(txt)


if __name__ == "__main__":
    test_diarized()
    test_no_diarize()
    test_formats()
    print("\nALL SMOKE CHECKS PASSED")
