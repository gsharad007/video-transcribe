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


def resolve_speaker_map(glossary: dict, speakers_arg: str | None) -> dict:
    """Resolve the effective speaker_map from a glossary + the --speakers CLI arg.

    `speakers_arg is None` (flag omitted) -> the glossary's own default map.
    `speakers_arg == ""` (flag passed explicitly empty) -> {} (no renaming at
    all), even if the glossary has a default -- lets a caller suppress a
    default meant for a different meeting instead of it silently applying to
    an unrelated one.
    `speakers_arg == "K=V,..."` -> exactly that map, replacing the default.
    """
    if speakers_arg is None:
        return glossary.get("speaker_map", {})
    speaker_map: dict[str, str] = {}
    for part in speakers_arg.split(","):
        key, sep, val = part.partition("=")
        if sep and val.strip():
            speaker_map[key.strip()] = val.strip()
    return speaker_map


def resolve_speaker_at(utterances: list[dict], specs: list[str] | None,
                       tolerance: float = 0.5) -> list[tuple[float, float, str]]:
    """Resolve --speaker-at "START=NAME" specs into [(range_start, range_end, name), ...]
    time ranges, one per spec, by matching START against the closest utterance's
    own start time (within `tolerance` seconds).

    This is the fix for a gap visual_id.py's merge_suspected workflow otherwise
    has no answer to: that check deliberately never auto-splits a flagged label
    (see its docstring -- a human must decide), but until now there was no way
    to actually APPLY that decision without hand-editing the transcript JSON.
    --speakers only renames a whole label; this overrides just the ONE
    utterance a human identified (e.g. from a merge_suspected report), leaving
    every other utterance under that label's existing name untouched.
    """
    specs = specs or []
    out: list[tuple[float, float, str]] = []
    for spec in specs:
        start_s, sep, name = spec.partition("=")
        if not sep or not name.strip():
            raise SystemExit(f"error: bad --speaker-at entry '{spec}'. Use START=NAME, "
                             "e.g. --speaker-at \"37.3=Sharad\" (START matches an "
                             "utterance's start time, e.g. from a visual-id .visual.json "
                             "report's \"utterance_start\")")
        t = float(start_s.strip())
        if not utterances:
            raise SystemExit(f"error: --speaker-at {spec!r} but the transcript has no "
                             "utterances to match against")
        closest = min(utterances, key=lambda u: abs(u["start"] - t))
        if abs(closest["start"] - t) > tolerance:
            raise SystemExit(f"error: --speaker-at {spec!r} matched no utterance within "
                             f"{tolerance}s (closest start: {closest['start']:.2f})")
        out.append((closest["start"], closest["end"], name.strip()))
    return out


def correct_conversation(data: dict, speaker_map: dict, compiled, *,
                         speaker_at: list[tuple[float, float, str]] | None = None,
                         ) -> tuple[Conversation, formats.Meta]:
    speaker_at = speaker_at or []

    def override_at(start: float, end: float) -> str | None:
        # A segment/utterance is "in" an overridden range if its start falls
        # inside it -- segments are narrower slices of their parent
        # utterance, so this covers both without needing separate matching.
        for r_start, r_end, name in speaker_at:
            if r_start - 1e-6 <= start <= r_end + 1e-6:
                return name
        return None

    def spk(s, start: float, end: float):
        override = override_at(start, end)
        if override is not None:
            return override
        return speaker_map.get(s, s) if s else s

    segments = [
        DiarizedSegment(s["id"], s["start"], s["end"], fix_text(s["text"], compiled),
                        spk(s.get("speaker"), s["start"], s["end"]))
        for s in data.get("segments", [])
    ]
    utterances = [
        Utterance(u["start"], u["end"], spk(u.get("speaker"), u["start"], u["end"]),
                  fix_text(u["text"], compiled))
        for u in data.get("utterances", [])
    ]
    # Preserve original first-appearance order; --speaker-at can introduce a
    # name not in the original diarized speakers list at all (e.g. splitting
    # a merge_suspected minority utterance out to a name of its own), so
    # append any such new names rather than only remapping existing ones.
    mapped_originals = [speaker_map.get(s, s) for s in data.get("speakers", [])]
    extra_names = [name for _, _, name in speaker_at if name not in mapped_originals]
    speakers = list(dict.fromkeys(mapped_originals + extra_names))

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
    p.add_argument("--glossary", type=Path, default=None,
                   help="JSON with 'speaker_map' and 'corrections'. Optional if you only "
                        "need --speaker-at (a pure per-utterance override needs no glossary).")
    p.add_argument("--speakers", default=None,
                   help="Override the glossary speaker map, e.g. "
                        "'Speaker 1=JV,Speaker 2=Sharad'. Pass '' explicitly to disable "
                        "the glossary's default map entirely (no renaming) -- e.g. when "
                        "the transcript's names are already resolved by voiceprints and "
                        "any leftover 'Speaker N' should stay generic rather than risk "
                        "matching a stale default meant for a different meeting.")
    p.add_argument("--speaker-at", action="append", default=None, metavar="START=NAME",
                   help="Override ONE utterance's speaker by its start time, e.g. "
                        "\"--speaker-at 37.3=Sharad\" (repeatable). This is the fix for a "
                        "visual_id.py merge_suspected finding: that check deliberately "
                        "never auto-splits a flagged label, so use this to apply your own "
                        "decision about which specific utterance actually belongs to "
                        "someone else, leaving every other utterance under that label's "
                        "existing name untouched. Takes priority over --speakers/the "
                        "glossary's whole-label map for the matched utterance.")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="output directory (default: next to the input)")
    p.add_argument("-f", "--format", dest="formats", action="append",
                   choices=sorted(EXT), metavar="FMT",
                   help="output format(s); default: txt srt json")
    args = p.parse_args(argv)

    glossary = json.loads(args.glossary.read_text(encoding="utf-8")) if args.glossary else {}
    compiled = compile_corrections(glossary.get("corrections", []))
    speaker_map = resolve_speaker_map(glossary, args.speakers)

    data = json.loads(args.input.read_text(encoding="utf-8"))
    speaker_at = resolve_speaker_at(data.get("utterances", []), args.speaker_at)
    conv, meta = correct_conversation(data, speaker_map, compiled, speaker_at=speaker_at)

    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in (args.formats or ["txt", "srt", "json"]):
        out = out_dir / (args.input.stem + EXT[fmt])
        out.write_text(formats.WRITERS[fmt](conv, meta), encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
