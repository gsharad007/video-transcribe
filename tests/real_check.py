"""Real-model check: faster-whisper word timestamps -> merge -> formats.

Transcribes the short SAPI sample with word_timestamps=True (no HF token needed),
then runs it through the no-diarize path and a synthetic 2-speaker split to
exercise the real Word objects through the speaker-grouping code.

Run: uv run python tests/real_check.py <sample.(mp4|wav)>
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from video_transcribe import audio, formats, merge
from video_transcribe.diarize import SpeakerTurn
from video_transcribe.transcribe import transcribe


def main(src: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        audio.extract_audio(src, wav)
        result = transcribe(wav, model_size="base", language="en", word_timestamps=True)

    nwords = sum(len(s.words) for s in result.segments)
    print(f"segments={len(result.segments)} words={nwords} dur={result.duration:.1f}s")
    assert nwords > 0, "expected per-word timestamps from faster-whisper"

    # Split the timeline in half into two fake speakers to drive the word->speaker path.
    half = result.duration / 2
    turns = [SpeakerTurn(0.0, half, "SPEAKER_00"),
             SpeakerTurn(half, result.duration + 1, "SPEAKER_01")]

    conv = merge.build_conversation(result.segments, turns, tidy=True)
    meta = formats.Meta.from_result(src.name, result, diarized=True)
    print(f"speakers={conv.speakers} utterances={len(conv.utterances)}")
    assert conv.speakers, "speakers should be labeled"
    assert any(u.speaker == "Speaker 1" for u in conv.utterances)

    print("\n----- diarized (mock 2-speaker) txt -----")
    print(formats.to_txt(conv, meta))

    conv2 = merge.build_conversation(result.segments, [], tidy=True)
    print("----- no-diarize txt -----")
    print(formats.to_txt(conv2, formats.Meta.from_result(src.name, result, diarized=False)))

    print("REAL CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
