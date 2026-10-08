"""Deterministic, model-free checks for the merge + format logic.

Run: uv run python tests/smoke.py
"""

from __future__ import annotations

from video_transcribe import formats, merge
from video_transcribe.correct import correct_conversation, resolve_speaker_at, resolve_speaker_map
from video_transcribe.diarize import SpeakerTurn
from video_transcribe.llm_correct import correct_texts_with_llm, diff_report
from video_transcribe.qc import check_transcript, find_phrase_repeats
from video_transcribe.transcribe import Segment, TranscriptionResult, Word
from video_transcribe.visual_id import (
    VisualReading, auto_name_from_visual, check_label_consistency, detect_border,
    fuzzy_match, label_crop_box, sample_points, vote_readings,
)


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


def test_resolve_speaker_map_explicit_empty_suppresses_default():
    # A meeting whose glossary default speaker_map is for a DIFFERENT recurring
    # meeting (e.g. "Speaker 1": "Mar" from an old 1-1 convention) must not
    # silently apply to some other meeting's unresolved "Speaker 1". Omitting
    # --speakers uses the default (the common, intended case); explicitly
    # passing "" must suppress it entirely -- this is the exact bug that
    # mislabeled an unrelated person as "Mar" in a group meeting transcript.
    glossary = {"speaker_map": {"Speaker 1": "Mar", "Speaker 2": "Sharad"}}

    assert resolve_speaker_map(glossary, None) == {"Speaker 1": "Mar", "Speaker 2": "Sharad"}
    assert resolve_speaker_map(glossary, "") == {}
    assert resolve_speaker_map(glossary, "Speaker 1=Ryan") == {"Speaker 1": "Ryan"}
    print("[ok] correct: --speakers '' suppresses the glossary default instead of using it")


def test_speaker_at_splits_one_utterance_out_of_a_label():
    # The exact real-world shape this exists for: a diarized label ("Amanda")
    # whose visual-id readings flagged one short utterance as actually a
    # different person -- --speaker-at applies a human's decision about that
    # ONE utterance without touching any other utterance under that label.
    utterances = [
        {"start": 37.3, "end": 39.18, "speaker": "Amanda", "text": "Chris, I think your mic is off."},
        {"start": 117.38, "end": 128.24, "speaker": "Amanda", "text": "Congrats JB."},
    ]
    ranges = resolve_speaker_at(utterances, ["37.3=Sharad"])
    assert ranges == [(37.3, 39.18, "Sharad")]

    data = {
        "utterances": utterances,
        "segments": [
            {"id": 0, "start": 37.3, "end": 39.18, "speaker": "Amanda", "text": "Chris, I think your mic is off."},
            {"id": 1, "start": 117.38, "end": 128.24, "speaker": "Amanda", "text": "Congrats JB."},
        ],
        "speakers": ["Amanda"],
    }
    conv, _meta = correct_conversation(data, {}, [], speaker_at=ranges)
    assert [u.speaker for u in conv.utterances] == ["Sharad", "Amanda"]
    assert [s.speaker for s in conv.segments] == ["Sharad", "Amanda"]
    assert set(conv.speakers) == {"Sharad", "Amanda"}

    try:
        resolve_speaker_at(utterances, ["99.0=Nobody"])
        assert False, "expected a SystemExit for an unmatched start time"
    except SystemExit:
        pass
    print("[ok] correct: --speaker-at splits one utterance out of a label without "
          "touching the rest")


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


def test_muxed_output_path():
    from pathlib import PurePosixPath

    from video_transcribe import audio

    # mp4 source -> .mkv output already differs, so no ".with-mic" suffix
    p = audio.muxed_output_path(PurePosixPath("/x/meeting.mp4"))
    assert p == PurePosixPath("/x/meeting.mkv"), p
    # mkv source -> same extension as output, so keep ".with-mic" to avoid clobber
    p = audio.muxed_output_path(PurePosixPath("/x/meeting.mkv"))
    assert p == PurePosixPath("/x/meeting.with-mic.mkv"), p
    # extension check is case-insensitive
    p = audio.muxed_output_path(PurePosixPath("/x/meeting.MKV"))
    assert p == PurePosixPath("/x/meeting.with-mic.mkv"), p
    # out_dir override lands the file elsewhere
    p = audio.muxed_output_path(PurePosixPath("/x/meeting.mp4"), PurePosixPath("/out"))
    assert p == PurePosixPath("/out/meeting.mkv"), p
    print("[ok] mux: output name drops '.with-mic' unless source is already .mkv")


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


def _draw_border(shape, bbox, color, thickness=4):
    """Synthetic frame with a stroke rectangle border, for detect_border tests."""
    import numpy as np
    frame = np.full(shape, 30, dtype=np.uint8)
    top, left, bottom, right = bbox
    t = thickness
    frame[top:top + t, left:right] = color
    frame[bottom - t:bottom, left:right] = color
    frame[top:bottom, left:left + t] = color
    frame[top:bottom, right - t:right] = color
    return frame


def test_detect_border():
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("[skip] detect_border: numpy not installed (uv sync --extra visual)")
        return

    # Border color/geometry measured from the real incident recording (see
    # visual_id.py's module docstring) -- NOT a vivid/saturated blue guess.
    # Using the real measured values here is what caught two real bugs during
    # development (a band width that scaled with tile size, and an
    # area-fraction floor sized for a filled region instead of a thin stroke)
    # that a purely synthetic "nice round numbers" test would have missed.
    frame = _draw_border((300, 300, 3), (40, 40, 260, 260), (166, 197, 247), thickness=4)
    det = detect_border(frame)
    assert det is not None, "should detect a real-shaped border"
    assert abs(det.bbox[0] - 40) <= 2 and abs(det.bbox[1] - 40) <= 2

    import numpy as np
    blank_frame = np.full((300, 300, 3), 30, dtype=np.uint8)
    assert detect_border(blank_frame) is None, "must not hallucinate a border from nothing"

    # A solid filled square (not a stroke) must be rejected even though it's
    # the same color -- this is the interior/perimeter shape check's whole job.
    filled = np.full((300, 300, 3), 30, dtype=np.uint8)
    filled[40:260, 40:260] = (166, 197, 247)
    assert detect_border(filled) is None, "a filled blob must not be mistaken for a stroke"
    print("[ok] visual_id: detect_border finds a real stroke, rejects blank/filled frames")


def test_fuzzy_match():
    # OCR reads a full name ("Sharad Gupta"); this project's roster convention
    # is first names only ("Sharad") -- whole-string difflib ratio alone
    # under-scores this real case (0.56, below any reasonable threshold) because
    # of the length mismatch from the surname. This was a real bug found by
    # testing against actual OCR output, not a hypothetical.
    assert fuzzy_match("Shared Gupta", ["Sharad", "Mar", "John"])[0] == "Sharad"
    assert fuzzy_match("Sharad Gupta", ["Sharad", "Mar", "John"]) == ("Sharad", 1.0)
    assert fuzzy_match("Amanda Diaz", ["Sharad", "Amanda", "John"])[0] == "Amanda"
    # no roster match at all -> reject, don't guess
    assert fuzzy_match("xyz123", ["Sharad", "Mar"]) == (None, 0.0)
    assert fuzzy_match("", ["Sharad"]) == (None, 0.0)
    assert fuzzy_match("Sharad", []) == (None, 0.0)
    print("[ok] visual_id: fuzzy_match handles first-name-vs-full-name OCR text")


def test_vote_readings():
    unanimous = vote_readings(["Mar", "Mar", "Mar"], utterance_start=0, utterance_end=5, label="Speaker 1")
    assert unanimous.name == "Mar" and unanimous.confidence == 1.0 and unanimous.coverage == 1.0

    tie = vote_readings(["Mar", "John", None], utterance_start=0, utterance_end=5, label="X")
    assert tie.name is None and tie.reject_reason == "ambiguous_vote"

    all_rejected = vote_readings([None, None, None], utterance_start=0, utterance_end=5, label="X")
    assert all_rejected.name is None and all_rejected.reject_reason == "no_border"

    too_short = vote_readings([], utterance_start=0, utterance_end=5, label="X")
    assert too_short.reject_reason == "insufficient_duration"
    print("[ok] visual_id: vote_readings (unanimous / tie / all-rejected / too-short)")


def test_sample_points():
    pts = sample_points(0, 20, spacing=2.0)
    assert 3 <= len(pts) <= 8
    assert all(0.3 <= p <= 19.7 for p in pts)
    assert sample_points(0, 0.4) == []  # too short even for the margin
    print("[ok] visual_id: sample_points spacing/margin/bounds")


def test_label_crop_box():
    box = label_crop_box((100, 100, 500, 500))  # left, top, right, bottom
    left, top, right, bottom = box
    assert left == 100 and right < 500 and top > 100 and bottom < 500
    print("[ok] visual_id: label_crop_box stays inside the tile bbox")


def _vreading(label, name, start, *, conf=1.0, n_accepted=3):
    return VisualReading(
        utterance_start=start, utterance_end=start + 3, label=label, name=name,
        confidence=conf, coverage=1.0, n_samples=3, n_accepted=n_accepted, reject_reason=None,
    )


def test_check_label_consistency():
    # Reproduces the actual 2026-08-12 incident's shape: a diarized label
    # ("Speaker 1") whose visual readings resolve to two different real
    # people -- one contributing exactly ONE utterance (the real minority
    # speaker had exactly one too). This is the single most important test
    # in this module.
    incident = [_vreading("Speaker 1", "Sharad", 0)]
    incident += [_vreading("Speaker 1", "Amanda", 10 + i * 10) for i in range(8)]
    incident += [_vreading("John", "John", 5)]
    flagged = check_label_consistency(incident)
    assert len(flagged) == 1 and flagged[0].label == "Speaker 1"
    assert set(flagged[0].names) == {"Sharad", "Amanda"}

    clean = [_vreading("Speaker 2", "Mar", i * 5) for i in range(5)]
    assert check_label_consistency(clean) == []

    # A single LOW-confidence misread on an otherwise-consistent label must
    # not manufacture a false alarm.
    low_conf = [_vreading("Speaker 3", "Ryan", i * 5) for i in range(9)]
    low_conf += [_vreading("Speaker 3", "Ness", 99, conf=0.4)]
    assert check_label_consistency(low_conf) == []

    # But a single HIGH-confidence, well-sampled one-off utterance for a 2nd
    # person SHOULD flag -- that's the whole point (see `incident` above).
    high_conf = [_vreading("Speaker 4", "Ryan", i * 5) for i in range(9)]
    high_conf += [_vreading("Speaker 4", "Ness", 99, conf=1.0)]
    assert len(check_label_consistency(high_conf)) == 1

    # The real false-alarm shape found during development: high "confidence"
    # (1 vote / 1 accepted sample = trivially 1.0) but only 1 accepted sample
    # out of many attempted -- a stray misdetected frame, not real evidence.
    # Every false merge_suspected flag hit while testing against real video
    # had exactly this shape; requiring >=2 agreeing samples fixes it.
    sparse_fluke = [_vreading("John", "John", i * 10, n_accepted=1) for i in range(2)]
    sparse_fluke += [_vreading("John", "Sharad", 99, conf=1.0, n_accepted=1)]
    assert check_label_consistency(sparse_fluke) == []

    # Two isolated single-sample flukes for the SAME wrong name, in two
    # DIFFERENT utterances, must not clear the count path just by reaching
    # raw utterance-count 2 -- each must individually be well-sampled.
    two_weak_utterances = [_vreading("Speaker 5", "Ryan", i * 10, n_accepted=1) for i in range(9)]
    two_weak_utterances += [
        _vreading("Speaker 5", "Ness", 90, n_accepted=1),
        _vreading("Speaker 5", "Ness", 95, n_accepted=1),
    ]
    assert check_label_consistency(two_weak_utterances) == []
    print("[ok] visual_id: check_label_consistency reproduces the real incident, "
          "rejects low-confidence and sparse-sample false alarms")


def test_auto_name_from_visual_never_renames_resolved_label():
    # A label already resolved (by voiceprint match or --track-speakers
    # ground truth) must NEVER be renamed by a visual-id guess, even a
    # clean unambiguous one -- only still-generic "Speaker N" labels are
    # fair game. This is the direct fix for the miscategorization the whole
    # feature exists to prevent.
    readings = [
        VisualReading(0, 3, "Mar", "SomeoneElse", 1.0, 1.0, 3, 3, None),
        VisualReading(10, 13, "Mar", "SomeoneElse", 1.0, 1.0, 3, 3, None),
        VisualReading(20, 23, "Speaker 1", "John", 1.0, 1.0, 3, 3, None),
    ]
    names = auto_name_from_visual(readings)
    assert "Mar" not in names
    assert names == {"Speaker 1": "John"}
    print("[ok] visual_id: auto_name_from_visual only renames still-generic labels")


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


def _seg(i, start, end, text, speaker="Sharad"):
    return {"id": i, "start": start, "end": end, "text": text, "speaker": speaker}


def test_qc_repeat_tiers():
    # The real Sync4 defect: one clause repeated 17x. Must be caught.
    loop = "and then I'll call it done " * 17
    hits = find_phrase_repeats(loop)
    assert hits and max(r for r, _n, _p in hits) >= 3, hits

    # Natural short disfluency must NOT reach the same bar. "yeah yeah yeah"
    # is 3 words total, under the min-run floor, so it isn't reported at all.
    assert find_phrase_repeats("yeah yeah yeah that works") == []
    print("[ok] qc: 17x clause loop detected, 3x one-word stutter ignored")


def test_qc_impossible_rate_needs_repetition():
    # Mild compression on coherent, non-repeating speech = timing only.
    # (Real case: Sync9, 19 words in 1.0s, content sensible and unique.)
    benign = {"segments": [
        _seg(0, 0.0, 30.0, "Something entirely different was discussed here at length."),
        _seg(1, 30.0, 31.0, "So I think I should be up and running by Thursday with "
                            "the AI stuff and then maybe."),
        _seg(2, 31.0, 60.0, "A completely unrelated closing remark follows now."),
    ]}
    kinds = {(f.kind, f.severity) for f in check_transcript(benign)}
    assert ("impossible_rate", "timing") in kinds, kinds
    assert ("impossible_rate", "hallucination") not in kinds, kinds

    # Same impossible rate, but echoing the neighbour verbatim = the real
    # defect (reproduces Sync12 ~400.7s: "binds things together" came back).
    echoing = {"segments": [
        _seg(0, 0.0, 30.0, "Which is another thing that kind of binds things together."),
        _seg(1, 30.0, 30.2, "Yeah I think it is a really good thing to do and it kind "
                            "of binds things together which is true."),
        _seg(2, 30.2, 60.0, "Anyway moving on to the next topic entirely."),
    ]}
    assert any(f.kind == "impossible_rate" and f.severity == "hallucination"
               for f in check_transcript(echoing))
    print("[ok] qc: impossible rate alone = timing; rate + echoed phrase = hallucination")


def test_qc_extreme_rate_and_duplicate_speaker():
    # 23 words in 0.04s (~575 w/s, real Engineering Meeting case): the segment
    # has no duration to hold its text, so it escalates without needing a repeat.
    extreme = {"segments": [
        _seg(0, 906.2, 906.24, "Going to have a lot of questions and you do not want "
                               "to be like oh I have no idea what to do here now"),
    ]}
    assert any(f.severity == "hallucination" for f in check_transcript(extreme))

    # Two raw labels resolving to one name inflate the header speaker count.
    dupes = {"segments": [], "utterances": [], "speakers": ["Mar", "Speaker 1", "Mar"]}
    found = [f for f in check_transcript(dupes) if f.kind == "duplicate_speaker"]
    assert len(found) == 1 and found[0].speaker == "Mar", found
    print("[ok] qc: extreme rate escalates alone; duplicate speaker name detected")


def test_qc_clean_transcript_has_no_review_findings():
    clean = {
        "speakers": ["Mar", "Sharad"],
        "utterances": [{"start": 0.0, "end": 8.0, "speaker": "Mar",
                        "text": "How is the material crash investigation going today?"}],
        "segments": [_seg(0, 0.0, 8.0, "How is the material crash investigation "
                                       "going today?", "Mar")],
    }
    assert [f for f in check_transcript(clean) if f.needs_review] == []
    print("[ok] qc: a clean transcript produces no review-level findings")


def test_pipeline_transcribe_argv():
    import argparse
    from pathlib import Path
    from video_transcribe.pipeline import Settings, transcribe_argv

    s = Settings(me="Sharad", glossary=Path("g.json"), voiceprints=Path("vp.json"),
                 voice_threshold=0.75, roster=["Mar", "Noah"])
    one = argparse.Namespace(video=Path("v.mp4"), mic=Path("m.m4a"), them="Mar", no_mux=False)
    # The exact shape run by hand for every biweekly sync.
    assert transcribe_argv("1on1", one, s) == [
        "v.mp4", "m.m4a", "--track-speakers", "Mar,Sharad", "--mux",
        "--hotwords-file", "g.json", "-f", "txt", "-f", "srt", "-f", "json"]

    hyb = argparse.Namespace(video=Path("v.mp4"), mic=Path("m.m4a"), speakers=None, no_mux=False)
    argv = transcribe_argv("group-hybrid", hyb, s)
    assert argv[:6] == ["v.mp4", "m.m4a", "--diarize-track", "0", "--track-speakers", "Sharad"]
    assert "--visual-id" in argv and argv[argv.index("--visual-roster") + 1] == "Mar,Noah"
    assert argv[argv.index("--voice-threshold") + 1] == "0.75"

    meet = argparse.Namespace(video=Path("v.mp4"), speakers=5)
    argv = transcribe_argv("group-meet", meet, s)
    assert argv[:2] == ["v.mp4", "--diarize"] and argv[argv.index("--speakers") + 1] == "5"
    assert "--mux" not in argv  # single file: nothing to mux
    print("[ok] pipeline: preset argv matches the hand-run commands")


def _fake_fix(i, frm, to, conf="high"):
    from types import SimpleNamespace
    return SimpleNamespace(utterance_index=i, from_text=frm, to_text=to,
                           reason="context", confidence=conf)


def test_pipeline_apply_fixes_scoped():
    from video_transcribe.pipeline import apply_fixes

    data = {
        "utterances": [
            {"start": 0.0, "end": 5.0, "speaker": "Mar", "text": "My Android phone is fine."},
            {"start": 5.0, "end": 9.0, "speaker": "Sharad",
             "text": "Satisfied with what Android provides right now."},
        ],
        "segments": [
            {"id": 0, "start": 0.0, "end": 5.0, "speaker": "Mar", "text": "My Android phone is fine."},
            {"id": 1, "start": 5.0, "end": 9.0, "speaker": "Sharad",
             "text": "Satisfied with what Android provides right now."},
        ],
    }
    log = apply_fixes(data, [
        _fake_fix(1, "Android", "Unreal"),            # applied, this utterance only
        _fake_fix(0, "phone", "device", "medium"),    # below threshold: suggestion only
        _fake_fix(1, "Horde", "Hoard"),               # text not in utterance: skipped
    ])
    assert data["utterances"][1]["text"] == "Satisfied with what Unreal provides right now."
    assert data["segments"][1]["text"] == "Satisfied with what Unreal provides right now."
    # Utterance 0's genuine "Android" must survive -- scoping is the point.
    assert data["utterances"][0]["text"] == "My Android phone is fine."
    assert log[0].startswith("applied") and log[1].startswith("suggested only")
    assert log[2].startswith("skipped (text not found")
    print("[ok] pipeline: LLM fixes apply per-utterance, high-confidence only")


def test_pipeline_llm_review_call_shape():
    from types import SimpleNamespace
    from video_transcribe.pipeline import llm_review

    seen = {}

    class FakeMessages:
        def __init__(self, stop_reason):
            self.stop_reason = stop_reason

        def parse(self, **kw):
            seen.update(kw)
            Review = kw["output_format"]
            parsed = Review(summary="s", notes=[], qc_verdicts=[], corrections=[
                {"utterance_index": 0, "from_text": "Android", "to_text": "Unreal",
                 "reason": "engine context", "confidence": "high"}])
            return SimpleNamespace(stop_reason=self.stop_reason, parsed_output=parsed)

    data = {"utterances": [{"start": 65.0, "end": 70.0, "speaker": "Sharad",
                            "text": "what Android provides"}]}
    out = llm_review(data, notes="* fog", findings=[], glossary=None,
                     model="claude-opus-5-5",
                     client=SimpleNamespace(messages=FakeMessages("end_turn")))
    assert out.corrections[0].to_text == "Unreal"
    assert seen["model"] == "claude-opus-5-5"
    assert "[0] 01:05 Sharad: what Android provides" in seen["messages"][0]["content"]

    try:
        llm_review(data, notes="", findings=[], glossary=None, model="m",
                   client=SimpleNamespace(messages=FakeMessages("refusal")))
    except RuntimeError as e:
        assert "refusal" in str(e)
    else:
        raise AssertionError("a refusal must not be read as an empty review")
    print("[ok] pipeline: llm_review builds the prompt, parses schema, surfaces refusals")


def test_pipeline_check_with_fake_llm():
    import argparse
    import json
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from video_transcribe import pipeline

    data = {
        "title": "t.mp4", "language": "en", "duration": 9.0, "model": "m", "diarized": True,
        "speakers": ["Mar", "Sharad"],
        "utterances": [{"start": 5.0, "end": 9.0, "speaker": "Sharad",
                        "text": "Satisfied with what Android provides right now."}],
        "segments": [{"id": 0, "start": 5.0, "end": 9.0, "speaker": "Sharad",
                      "text": "Satisfied with what Android provides right now."}],
    }

    def fake_review(data, *, notes, findings, glossary, model):
        assert "volumetric fog" in notes
        return SimpleNamespace(
            summary="ok",
            notes=[SimpleNamespace(note="volumetric fog", status="confirmed",
                                   utterance_index=0, evidence="what ... provides")],
            corrections=[_fake_fix(0, "Android", "Unreal")],
            qc_verdicts=[])

    real = pipeline.llm_review
    pipeline.llm_review = fake_review
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            cfg = Path(tmp) / "cfg.json"
            cfg.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(transcript=path, notes=None,
                                      notes_text="* volumetric fog", llm=True,
                                      apply_llm_fixes=True, llm_model=None,
                                      config=cfg, glossary=None)
            rc = pipeline.run("check", args)
            assert rc == pipeline.EXIT_CLEAN, rc
            assert "Unreal provides" in (Path(tmp) / "t.txt").read_text(encoding="utf-8")
            report = (Path(tmp) / "t.report.md").read_text(encoding="utf-8")
            assert "confirmed" in report and "applied" in report
    finally:
        pipeline.llm_review = real
    print("[ok] pipeline: check + LLM review writes report and applies fix end-to-end")


if __name__ == "__main__":
    test_diarized()
    test_no_diarize()
    test_resolve_speaker_map_explicit_empty_suppresses_default()
    test_speaker_at_splits_one_utterance_out_of_a_label()
    test_voice_names_override()
    test_muxed_output_path()
    test_hybrid_diarize_plus_track()
    test_llm_correct()
    test_llm_correct_batches()
    test_llm_correct_long_utterance_batch()
    test_llm_correct_dropped_index()
    test_voiceprint_store()
    test_voiceprint_exclusive_assignment()
    test_detect_border()
    test_fuzzy_match()
    test_vote_readings()
    test_sample_points()
    test_label_crop_box()
    test_check_label_consistency()
    test_auto_name_from_visual_never_renames_resolved_label()
    test_qc_repeat_tiers()
    test_qc_impossible_rate_needs_repetition()
    test_qc_extreme_rate_and_duplicate_speaker()
    test_qc_clean_transcript_has_no_review_findings()
    test_pipeline_transcribe_argv()
    test_pipeline_apply_fixes_scoped()
    test_pipeline_llm_review_call_shape()
    test_pipeline_check_with_fake_llm()
    test_formats()
    print("\nALL SMOKE CHECKS PASSED")
