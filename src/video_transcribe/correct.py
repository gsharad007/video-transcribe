"""Apply a glossary (term corrections + speaker names) to a finished transcript.

Post-processing only -- no re-transcription. Reads a transcript JSON produced by
this tool, fixes ASR mis-spellings of names/keywords (word-boundary, case-
insensitive) and relabels speakers, then re-emits txt / srt / vtt / json.

The glossary file is supplied by the caller (kept out of this repo), shaped:
  {
    "speaker_map": {"Speaker 1": "Mar", "Speaker 2": "Sharad"},
    "corrections": [{"from": "grim beaker", "to": "GrimeReaper"}, ...]
  }

Usage:
  uv run python -m video_transcribe.correct TRANSCRIPT.json --glossary g.json -o OUTDIR
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from video_transcribe import formats
from video_transcribe.merge import Conversation, DiarizedSegment, Utterance

EXT = {"txt": ".txt", "srt": ".srt", "vtt": ".vtt", "json": ".json"}


def compile_corrections(corrections: list[dict]) -> list[tuple[re.Pattern, str]]:
    """Longest 'from' first (so multi-word terms win), word-boundary, case-insensitive."""
    ordered = sorted(corrections, key=lambda c: -len(c["from"]))
    return [(re.compile(rf"\b{re.escape(c['from'])}\b", re.IGNORECASE), c["to"]) for c in ordered]


def fix_text(text: str, compiled: list[tuple[re.Pattern, str]]) -> str:
    for rx, to in compiled:
        text = rx.sub(to, text)
    return text


def correct_conversation(data: dict, speaker_map: dict, compiled) -> tuple[Conversation, formats.Meta]:
    def spk(s):
        return speaker_map.get(s, s) if s else s

    segments = [
        DiarizedSegment(s["id"], s["start"], s["end"], fix_text(s["text"], compiled), spk(s.get("speaker")))
        for s in data.get("segments", [])
    ]
    utterances = [
        Utterance(u["start"], u["end"], spk(u.get("speaker")), fix_text(u["text"], compiled))
        for u in data.get("utterances", [])
    ]
    speakers = [speaker_map.get(s, s) for s in data.get("speakers", [])]

    conv = Conversation(segments=segments, utterances=utterances, speakers=speakers)
    meta = formats.Meta(
        title=data.get("title", "transcript"),
        language=data.get("language", "?"),
        duration=float(data.get("duration", 0.0)),
        model=data.get("model", "?"),
        diarized=bool(data.get("diarized")),
    )
    return conv, meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="video-transcribe-correct",
        description="Apply glossary term-corrections + speaker names to a finished "
                    "transcript JSON (no re-transcription).",
    )
    p.add_argument("input", type=Path, help="transcript .json produced by video-transcribe")
    p.add_argument("--glossary", type=Path, required=True,
                   help="JSON with 'speaker_map' and 'corrections'")
    p.add_argument("--speakers", default=None,
                   help="Override the glossary speaker map, e.g. "
                        "'Speaker 1=JV,Speaker 2=Sharad'")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="output directory (default: next to the input)")
    p.add_argument("-f", "--format", dest="formats", action="append",
                   choices=sorted(EXT), metavar="FMT",
                   help="output format(s); default: txt srt json")
    args = p.parse_args(argv)

    glossary = json.loads(args.glossary.read_text(encoding="utf-8"))
    compiled = compile_corrections(glossary.get("corrections", []))
    speaker_map = glossary.get("speaker_map", {})
    if args.speakers:
        speaker_map = {}
        for part in args.speakers.split(","):
            key, sep, val = part.partition("=")
            if sep and val.strip():
                speaker_map[key.strip()] = val.strip()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    conv, meta = correct_conversation(data, speaker_map, compiled)

    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in (args.formats or ["txt", "srt", "json"]):
        out = out_dir / (args.input.stem + EXT[fmt])
        out.write_text(formats.WRITERS[fmt](conv, meta), encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
