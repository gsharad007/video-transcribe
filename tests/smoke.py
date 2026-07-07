"""Deterministic, model-free checks for the merge + format logic.

Run: uv run python tests/smoke.py
"""

from __future__ import annotations

from video_transcribe import formats, merge
from video_transcribe.diarize import SpeakerTurn
from video_transcribe.llm_correct import correct_texts_with_llm, diff_report
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


def test_voice_names_override():
    # A confident voiceprint match should replace the generic "Speaker N" label;
    # any raw label *not* in the match dict still falls back to "Speaker N",
    # numbered only among the unmatched.
    conv = merge.build_conversation(make_segments(), TURNS, tidy=True,
                                    voice_names={"SPEAKER_01": "Mar"})
    assert conv.speakers == ["Mar", "Speaker 1"], conv.speakers
    assert conv.utterances[0].speaker == "Mar"
    assert conv.utterances[1].speaker == "Speaker 1"

    group = merge.diarized_track(RESULT, TURNS, voice_names={"SPEAKER_00": "Ness"})
    speakers = [spk for _, spk in group]
    assert speakers == ["Speaker 1", "Ness"], speakers
    print("[ok] merge: voiceprint-matched labels override generic 'Speaker N'")


def test_hybrid_diarize_plus_track():
    # Group track (diarized, 2 speakers) + a separate fixed-speaker mic track,
    # merged by timestamp -- e.g. a meeting recording + your own mic.
    group = merge.diarized_track(RESULT, TURNS)
    mic_segments = [
        Segment(0, 1.0, 2.0, " quick aside", ()),
    ]
    mic = [(s, "Sharad") for s in merge.clean_segments(mic_segments)]
    conv = merge.build_conversation_from_tagged([group, mic], tidy=True)

    assert conv.speakers == ["Speaker 1", "Sharad", "Speaker 2"], conv.speakers
    assert len(conv.utterances) == 3, conv.utterances
    # interleaved by start time: Speaker 1 [0,3), Sharad [1,2) sorts after by
    # (start, end) tie-break only when starts match -- here Sharad's segment
    # starts inside Speaker 1's utterance span, so it should land second.
    assert [u.speaker for u in conv.utterances] == ["Speaker 1", "Sharad", "Speaker 2"]
    print("[ok] hybrid merge: diarized track + fixed-speaker track, timestamp order")


class _FakeParsed:
    def __init__(self, corrections):
        self.parsed_output = type("B", (), {"corrections": corrections})()


class _FakeMessages:
    def __init__(self, transform):
        self.transform = transform
        self.calls = 0

    def parse(self, *, model, max_tokens, system, messages, output_format):
        self.calls += 1
        Correction = type("C", (), {})
        corrections = []
        for line in messages[0]["content"].splitlines():
            idx_str, sep, text = line.partition(":")
            if not sep or not idx_str.strip().isdigit():
                continue
            c = Correction()
            c.index = int(idx_str.strip())
            c.text = self.transform(c.index, text.strip())
            corrections.append(c)
        return _FakeParsed(corrections)


class _FakeClient:
    def __init__(self, transform):
        self.messages = _FakeMessages(transform)


def test_llm_correct():
    try:
        import pydantic  # noqa: F401
    except ImportError:
        print("[skip] llm_correct: pydantic not installed (uv sync --extra llm)")
        return

    texts = ["hello wrold", "my name is grim beaker", "unchanged text"]
    fixes = {0: "hello world", 1: "my name is GrimeReaper"}
    client = _FakeClient(lambda idx, text: fixes.get(idx, text))

    corrected = correct_texts_with_llm(texts, client=client)
    assert corrected == ["hello world", "my name is GrimeReaper", "unchanged text"]
    assert client.messages.calls == 1

    report = diff_report(texts, corrected, starts=[0.0, 5.0, 10.0], speakers=["Mar", None, "Mar"])
    assert "hello wrold" in report and "hello world" in report
    assert "unchanged text" not in report
    print("[ok] llm_correct: single-batch correction + diff report")


def test_llm_correct_batches():
    try:
        import pydantic  # noqa: F401
    except ImportError:
        print("[skip] llm_correct_batches: pydantic not installed (uv sync --extra llm)")
        return

    n = 130  # > the internal per-batch item cap (60) -- forces three API calls
    texts = [f"line {i}" for i in range(n)]
    client = _FakeClient(lambda idx, text: "CHANGED" if idx in (0, n - 1) else text)

    corrected = correct_texts_with_llm(texts, client=client)
    assert len(corrected) == n
    assert corrected[0] == "CHANGED" and corrected[-1] == "CHANGED"
    assert corrected[1] == texts[1]
    assert client.messages.calls == 3
    print("[ok] llm_correct: multi-batch index alignment")


def test_llm_correct_long_utterance_batch():
    try:
        import pydantic  # noqa: F401
    except ImportError:
        print("[skip] llm_correct_long_utterance_batch: pydantic not installed (uv sync --extra llm)")
        return

    # A few long utterances should split into their own batches by character
    # budget, not get packed together into one oversized request.
    long_text = " ".join(["word"] * 4000)
    texts = [long_text, "short one", long_text, long_text]
    client = _FakeClient(lambda idx, text: text)

    corrected = correct_texts_with_llm(texts, client=client)
    assert corrected == texts
    assert client.messages.calls >= 3, client.messages.calls
    print("[ok] llm_correct: long utterances split across batches by char budget")


def test_llm_correct_dropped_index():
    try:
        import pydantic  # noqa: F401
    except ImportError:
        print("[skip] llm_correct_dropped_index: pydantic not installed (uv sync --extra llm)")
        return

    texts = ["alpha", "beta", "gamma"]

    # simulate the model silently omitting index 1 from its response
    class _SkippingMessages(_FakeMessages):
        def parse(self, *, model, max_tokens, system, messages, output_format):
            result = super().parse(model=model, max_tokens=max_tokens, system=system,
                                   messages=messages, output_format=output_format)
            result.parsed_output.corrections = [
                c for c in result.parsed_output.corrections if c.index != 1
            ]
            return result

    client = _FakeClient(lambda idx, text: "FIXED" if idx == 0 else text)
    client.messages = _SkippingMessages(lambda idx, text: "FIXED" if idx == 0 else text)
    corrected = correct_texts_with_llm(texts, client=client)
    assert corrected == ["FIXED", "beta", "gamma"], corrected  # index 1 falls back to original
    print("[ok] llm_correct: falls back to original text when the model drops an index")


def test_voiceprint_store():
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("[skip] voiceprint_store: numpy not installed (uv sync --extra diarize)")
        return

    import tempfile as _tempfile
    from pathlib import Path as _Path

    from video_transcribe.voiceprint import VoiceprintStore

    store = VoiceprintStore(path=_Path("unused.json"))
    # three well-separated synthetic "voices" in 3-D
    store.add("Ryan", [1.0, 0.0, 0.0])
    store.add("Ryan", [0.98, 0.02, 0.0])  # a second, slightly-noisy sample
    store.add("Mar", [0.0, 1.0, 0.0])

    name, score = store.match([0.99, 0.01, 0.0])
    assert name == "Ryan" and score > 0.9, (name, score)

    name, score = store.match([0.0, 0.99, 0.01])
    assert name == "Mar", (name, score)

    # something far from both known voices shouldn't clear the threshold
    name, score = store.match([0.0, 0.0, 1.0], threshold=0.75)
    assert name is None, (name, score)

    with _tempfile.TemporaryDirectory() as tmp:
        path = _Path(tmp) / "voiceprints.json"
        store.path = path
        store.save()
        reloaded = VoiceprintStore.load(path)
        assert set(reloaded.people) == {"Ryan", "Mar"}
        assert len(reloaded.people["Ryan"]) == 2

    print("[ok] voiceprint: store add/match/centroid + JSON round-trip")


def test_voiceprint_exclusive_assignment():
    try:
        import numpy as np
    except ImportError:
        print("[skip] voiceprint_exclusive: numpy not installed (uv sync --extra diarize)")
        return

    from video_transcribe.voiceprint import VoiceprintStore, identify_turns

    # The measured "attractor" scenario: cluster A is clearly P1; cluster B
    # also scores highest against P1 (0.8 vs 0.6 for its true speaker P2).
    # Independent argmax would give P1 both clusters; exclusive assignment
    # must give A->P1 (stronger claim) and B->P2.
    store = VoiceprintStore(path=None)
    store.add("P1", [1.0, 0.0, 0.0])
    store.add("P2", [0.0, 1.0, 0.0])

    sr = 16000
    # waveform regions encode which cluster a crop belongs to (value 1 vs 2)
    waveform = np.concatenate([np.full((1, 2 * sr), 1.0), np.full((1, 2 * sr), 2.0)], axis=1)
    cluster_vecs = {1.0: np.array([1.0, 0.05, 0.0]),      # ~P1
                    2.0: np.array([0.8, 0.6, 0.0])}       # closer to P1 than to P2!

    def fake_embedder(d):
        return cluster_vecs[float(np.asarray(d["waveform"]).mean())]

    turns = [SpeakerTurn(0.0, 2.0, "SPEAKER_00"), SpeakerTurn(2.0, 4.0, "SPEAKER_01")]
    result = identify_turns(turns, waveform, sr, fake_embedder, store, threshold=0.5)
    assert result == {"SPEAKER_00": "P1", "SPEAKER_01": "P2"}, result

    # below-threshold clusters stay unassigned rather than grabbing a leftover
    result = identify_turns(turns, waveform, sr, fake_embedder, store, threshold=0.9)
    assert result == {"SPEAKER_00": "P1"}, result
    print("[ok] voiceprint: exclusive one-to-one assignment beats the attractor")


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
    test_voice_names_override()
    test_hybrid_diarize_plus_track()
    test_llm_correct()
    test_llm_correct_batches()
    test_llm_correct_long_utterance_batch()
    test_llm_correct_dropped_index()
    test_voiceprint_store()
    test_voiceprint_exclusive_assignment()
    test_formats()
    print("\nALL SMOKE CHECKS PASSED")
