"""LLM-assisted correction pass over a finished transcript, via the Claude API.

Strictly a correction pass: fixes misheard names/jargon (optionally against a
glossary) and obvious ASR errors, one-to-one with the input utterances -- it
never rewrites, summarizes, or invents content. Writes to *separate* `.llm.*`
files (the original transcript is untouched) plus a `.llm-changes.txt` diff so
the corrections are visible without eyeballing two full transcripts.

Only utterance-level text (the paragraph grain used in .txt/.json) is
corrected -- segments (the subtitle-cue grain used for srt/vtt) are not
touched, so output is restricted to txt/json.

Unlike the rest of this pipeline, this sends transcript text -- names, meeting
content -- to Anthropic's API. Costs roughly a few cents per meeting on
claude-opus-4-8.

Usage:
  uv run python -m video_transcribe.llm_correct TRANSCRIPT.json [--glossary g.json] [--model claude-opus-4-8]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from video_transcribe import formats
from video_transcribe.correct import correct_conversation

EXT = {"txt": ".txt", "json": ".json"}
# Batches are capped by BOTH item count and total input characters -- some
# utterances are long monologues (thousands of characters), and the corrected
# output echoes back roughly that much text per item plus JSON overhead, so a
# batch sized by item count alone can blow past max_tokens and truncate mid-JSON.
_BATCH_MAX_ITEMS = 60
_BATCH_MAX_CHARS = 15_000
_MAX_TOKENS = 12_000
_DEFAULT_MODEL = "claude-opus-4-8"

_SYSTEM = """You are proofreading an automatic speech-recognition (ASR) transcript. \
You will be given a numbered list of utterances. For each one, fix ONLY:
- misheard proper nouns, names, and jargon (a glossary of correct terms may be provided)
- obvious ASR mistakes (wrong homophone, garbled word, wrong word boundary)

Do NOT:
- rewrite, rephrase, summarize, or "clean up" phrasing
- change meaning, add words, or remove words, except to fix a clear ASR error
- merge, split, reorder, or drop any utterance
- change punctuation or casing except where it's part of fixing a name/term

Return exactly one entry per input index, in the same order, even when no \
correction is needed -- in that case return the text unchanged. Never skip an index."""


def _load_glossary_context(path: Path | None) -> str:
    if path is None:
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    if terms := data.get("hotwords", []):
        lines.append("Known correct names/terms: " + ", ".join(terms))
    if corrections := data.get("corrections", []):
        pairs = ", ".join(f"{c['from']} -> {c['to']}" for c in corrections)
        lines.append("Known ASR mistakes and their fixes: " + pairs)
    return "\n".join(lines)


def _batch_ranges(texts: list[str], *, max_items: int, max_chars: int) -> Iterable[tuple[int, int]]:
    """Yield (start, count) ranges sized by item count AND total character budget.

    A single utterance longer than `max_chars` still gets its own batch --
    always makes progress rather than looping forever.
    """
    start, n = 0, len(texts)
    while start < n:
        count, chars = 0, 0
        while start + count < n and count < max_items:
            length = len(texts[start + count])
            if count > 0 and chars + length > max_chars:
                break
            chars += length
            count += 1
        count = max(count, 1)
        yield start, count
        start += count


def correct_texts_with_llm(
    texts: list[str], *, glossary_context: str = "", model: str = _DEFAULT_MODEL, client=None,
) -> list[str]:
    """Correct a list of ASR texts via Claude, 1:1 aligned with the input.

    Pass `client` (an `anthropic.Anthropic`-shaped object) to reuse a client or
    inject a fake one for testing; otherwise one is constructed on demand.
    """
    try:
        from pydantic import BaseModel
    except ImportError as e:
        raise RuntimeError("the 'llm' extra is required: uv sync --extra llm") from e

    class _Correction(BaseModel):
        index: int
        text: str

    class _CorrectionBatch(BaseModel):
        corrections: list[_Correction]

    if client is None:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("the 'llm' extra is required: uv sync --extra llm") from e
        client = anthropic.Anthropic()
    corrected: list[str | None] = [None] * len(texts)

    for batch_start, count in _batch_ranges(texts, max_items=_BATCH_MAX_ITEMS, max_chars=_BATCH_MAX_CHARS):
        batch = texts[batch_start:batch_start + count]
        numbered = "\n".join(f"{batch_start + i}: {t}" for i, t in enumerate(batch))
        user_content = f"{glossary_context}\n\n{numbered}" if glossary_context else numbered

        response = client.messages.parse(
            model=model,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            output_format=_CorrectionBatch,
        )
        by_index = {c.index: c.text for c in response.parsed_output.corrections}
        for i in range(batch_start, batch_start + len(batch)):
            if i not in by_index:
                print(f"warning: model dropped utterance index {i}; keeping original text",
                      file=sys.stderr)
                corrected[i] = texts[i]
            else:
                corrected[i] = by_index[i]

    return corrected  # type: ignore[return-value]


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(max(0.0, seconds)), 60)
    return f"{m:02d}:{s:02d}"


def diff_report(originals: list[str], corrected: list[str], *, starts: list[float],
                speakers: list[str | None]) -> str:
    lines = []
    for orig, new, start, speaker in zip(originals, corrected, starts, speakers):
        if new.strip() == orig.strip():
            continue
        who = f"{speaker}: " if speaker else ""
        lines.append(f"[{_fmt_ts(start)}] {who}\n  - {orig}\n  + {new}")
    if not lines:
        return "No corrections made.\n"
    return "\n\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="video-transcribe-llm-correct",
        description="LLM-assisted correction pass (Claude API) over a finished "
                    "transcript JSON. Writes separate .llm.txt/.llm.json files "
                    "plus a .llm-changes.txt diff -- never overwrites the original.",
    )
    p.add_argument("input", type=Path, help="transcript .json produced by video-transcribe")
    p.add_argument("--glossary", type=Path, default=None,
                   help="optional glossary JSON ('hotwords' / 'corrections') for context")
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help=f"Claude model (default: {_DEFAULT_MODEL})")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="output directory (default: next to the input)")
    p.add_argument("-f", "--format", dest="formats", action="append",
                   choices=sorted(EXT), metavar="FMT",
                   help="output format(s) for the corrected transcript; default: txt json")
    args = p.parse_args(argv)

    data = json.loads(args.input.read_text(encoding="utf-8"))
    conv, meta = correct_conversation(data, speaker_map={}, compiled=[])
    glossary_context = _load_glossary_context(args.glossary)

    print(f"correcting {len(conv.utterances)} utterances with {args.model} ...", file=sys.stderr)
    originals = [u.text for u in conv.utterances]
    corrected = correct_texts_with_llm(originals, glossary_context=glossary_context, model=args.model)

    new_conv = replace(conv, utterances=[
        replace(u, text=t) for u, t in zip(conv.utterances, corrected)
    ])

    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem + ".llm"
    for fmt in (args.formats or ["txt", "json"]):
        out = out_dir / (stem + EXT[fmt])
        out.write_text(formats.WRITERS[fmt](new_conv, meta), encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)

    report = diff_report(
        originals, corrected,
        starts=[u.start for u in conv.utterances],
        speakers=[u.speaker for u in conv.utterances],
    )
    changes_path = out_dir / (args.input.stem + ".llm-changes.txt")
    changes_path.write_text(report, encoding="utf-8")
    n_changed = sum(1 for o, c in zip(originals, corrected) if c.strip() != o.strip())
    print(f"wrote {changes_path} ({n_changed} of {len(originals)} utterances changed)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
