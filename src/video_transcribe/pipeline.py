"""One command for the whole recurring job: transcribe -> clean -> check -> (LLM review).

Every meeting used to be five hand-chained steps: transcribe, glossary pass,
QC scan, visual-ID review for group calls, then reading the transcript against
the user's notes and spotting leftover mis-hearings. This runs them in order
from one preset and writes one report.

Presets:
  1on1          VIDEO MIC   -- screen recording + your own mic, exact speakers
  group-hybrid  VIDEO MIC   -- diarize the call video, your mic fixed, + voiceprints + visual ID
  group-meet    VIDEO       -- diarize one recording + voiceprints + visual ID
  check         TRANSCRIPT  -- the post-transcription steps only, on an existing .json

The optional LLM step (``--llm``) sends the transcript and your notes to the
Claude API. It is off by default. What it proposes -- note coverage, suspected
mis-hearings -- goes into the report; fixes touch the transcript only with
``--apply-llm-fixes``, and only fixes the model rated high-confidence, each
scoped to the single utterance it was found in.

Defaults (glossary path, voiceprint store, your name, roster) come from
``~/.video-transcribe/config.json`` so a run is just the preset, the files, and
the other person's name. Command-line flags override the file.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from video_transcribe import qc

CONFIG_PATH = Path.home() / ".video-transcribe" / "config.json"
DEFAULT_LLM_MODEL = "claude-opus-5-5"
_FORMATS = ["-f", "txt", "-f", "srt", "-f", "json"]

# Exit codes: what a caller (or the TUI's run-state colour) needs to know.
EXIT_CLEAN, EXIT_REVIEW, EXIT_FAILED = 0, 1, 2


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@dataclass
class Settings:
    me: str = "Sharad"
    glossary: Path | None = None
    voiceprints: Path | None = None
    voice_threshold: float = 0.75
    roster: list[str] = field(default_factory=list)
    llm_model: str = DEFAULT_LLM_MODEL


def load_settings(path: Path | None, args: argparse.Namespace) -> Settings:
    """Config file first, then any flag the user actually passed on top."""
    s = Settings()
    path = path or CONFIG_PATH
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        s.me = data.get("me", s.me)
        s.glossary = Path(data["glossary"]) if data.get("glossary") else None
        s.voiceprints = Path(data["voiceprints"]) if data.get("voiceprints") else None
        s.voice_threshold = float(data.get("voice_threshold", s.voice_threshold))
        s.roster = list(data.get("roster", []))
        s.llm_model = data.get("llm_model", s.llm_model)
    if getattr(args, "me", None):
        s.me = args.me
    if getattr(args, "glossary", None):
        s.glossary = args.glossary
    if getattr(args, "voiceprints", None):
        s.voiceprints = args.voiceprints
    if getattr(args, "roster", None):
        s.roster = [n.strip() for n in args.roster.split(",") if n.strip()]
    if getattr(args, "llm_model", None):
        s.llm_model = args.llm_model
    return s


# --------------------------------------------------------------------------- #
# transcribe-argv builders (pure, unit-tested)
# --------------------------------------------------------------------------- #

def transcribe_argv(preset: str, args: argparse.Namespace, s: Settings) -> list[str]:
    hot = ["--hotwords-file", str(s.glossary)] if s.glossary else []
    mux = [] if getattr(args, "no_mux", False) else ["--mux"]
    count = ["--speakers", str(args.speakers)] if getattr(args, "speakers", None) else []
    voice = (["--voiceprints", str(s.voiceprints), "--voice-threshold", str(s.voice_threshold)]
             if s.voiceprints else [])
    visual = ["--visual-id"] + (["--visual-roster", ",".join(s.roster)] if s.roster else [])

    if preset == "1on1":
        return [str(args.video), str(args.mic), "--track-speakers", f"{args.them},{s.me}",
                *mux, *hot, *_FORMATS]
    if preset == "group-hybrid":
        return [str(args.video), str(args.mic), "--diarize-track", "0",
                "--track-speakers", s.me, *count, *voice, *visual, *mux, *hot, *_FORMATS]
    if preset == "group-meet":
        return [str(args.video), "--diarize", *count, *voice, *visual, *hot, *_FORMATS]
    raise ValueError(f"unknown preset {preset!r}")


# --------------------------------------------------------------------------- #
# LLM review
# --------------------------------------------------------------------------- #

_REVIEW_SYSTEM = """You are checking an automatic speech-recognition transcript of a work \
meeting against the attendee's own notes. You do three things:

1. For each note, decide whether the transcript supports it. Cite the utterance index and \
quote a short piece of evidence. Use status "off_record" only when the note itself says \
it was not discussed in the meeting.
2. List words that look like ASR mis-hearings: a wrong name, product, or technical term, \
judged from context and the known-terms list. Give the exact text as it appears in that \
utterance and the replacement. Do not propose rephrasing, grammar fixes, filler removal, \
or style changes -- only clear recognition errors. Rate confidence "high" only when the \
context makes the intended word unambiguous.
3. For each automated quality finding given to you, say whether the text around it reads \
as a real problem or as harmless.

Be literal and conservative. An empty corrections list is a normal, good outcome."""


def _review_models():
    from pydantic import BaseModel

    class NoteCheck(BaseModel):
        note: str
        status: Literal["confirmed", "partial", "not_found", "off_record"]
        utterance_index: int | None
        evidence: str

    class Fix(BaseModel):
        utterance_index: int
        from_text: str
        to_text: str
        reason: str
        confidence: Literal["high", "medium", "low"]

    class QcVerdict(BaseModel):
        finding: str
        verdict: Literal["real_problem", "harmless", "unclear"]
        comment: str

    class Review(BaseModel):
        notes: list[NoteCheck]
        corrections: list[Fix]
        qc_verdicts: list[QcVerdict]
        summary: str

    return Review


def _transcript_for_prompt(data: dict) -> str:
    lines = []
    for i, u in enumerate(data.get("utterances", [])):
        m, sec = divmod(int(u.get("start", 0)), 60)
        lines.append(f"[{i}] {m:02d}:{sec:02d} {u.get('speaker') or '?'}: {u.get('text', '')}")
    return "\n".join(lines)


def llm_review(data: dict, *, notes: str, findings: list[qc.Finding],
               glossary: Path | None, model: str, client=None):
    """One structured-output call. Returns the parsed Review (pydantic model).

    ``client`` is injectable so tests can run without network or credentials.
    """
    Review = _review_models()
    if client is None:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("the 'llm' extra is required "
                               "(uv sync with --extra llm alongside your other extras)") from e
        client = anthropic.Anthropic()

    known = ""
    if glossary and glossary.exists():
        g = json.loads(glossary.read_text(encoding="utf-8"))
        known = "Known correct names/terms: " + ", ".join(g.get("hotwords", []))
    qc_text = "\n".join(
        f"- {f.severity} at {f.start if f.start is not None else '-'}s: {f.detail}"
        for f in findings) or "(none)"
    user = (f"{known}\n\nATTENDEE NOTES:\n{notes or '(no notes given)'}\n\n"
            f"AUTOMATED QUALITY FINDINGS:\n{qc_text}\n\n"
            f"TRANSCRIPT (index, time, speaker, text):\n{_transcript_for_prompt(data)}")

    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=_REVIEW_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_format=Review,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined the review request (stop_reason=refusal)")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("review output hit max_tokens before finishing")
    return response.parsed_output


def apply_fixes(data: dict, fixes, *, min_confidence: str = "high") -> list[str]:
    """Apply fixes in place, each limited to its own utterance (and that
    utterance's segments). Returns one log line per proposed fix.

    Scoping matters: "Android" may be a real word elsewhere in the meeting; a
    global replace would rewrite every occurrence on the strength of one
    context.
    """
    rank = {"low": 0, "medium": 1, "high": 2}
    utterances = data.get("utterances", [])
    log = []
    for fx in fixes:
        tag = f"[{fx.utterance_index}] {fx.from_text!r} -> {fx.to_text!r} ({fx.confidence})"
        if rank[fx.confidence] < rank[min_confidence]:
            log.append(f"suggested only  {tag}: {fx.reason}")
            continue
        if not 0 <= fx.utterance_index < len(utterances):
            log.append(f"skipped (bad index) {tag}")
            continue
        u = utterances[fx.utterance_index]
        rx = re.compile(rf"\b{re.escape(fx.from_text)}\b", re.IGNORECASE)
        if not rx.search(u.get("text", "")):
            log.append(f"skipped (text not found in utterance) {tag}")
            continue
        u["text"] = rx.sub(fx.to_text, u["text"])
        for seg in data.get("segments", []):
            if u["start"] - 1e-6 <= seg.get("start", -1) <= u["end"] + 1e-6:
                seg["text"] = rx.sub(fx.to_text, seg.get("text", ""))
        log.append(f"applied         {tag}: {fx.reason}")
    return log


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def _capture(fn, *a, **kw) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a, **kw)
    return rc, buf.getvalue()


def _render(json_path: Path) -> None:
    """Re-emit txt/srt/json from the (possibly edited) .json."""
    from video_transcribe import correct
    with contextlib.redirect_stderr(io.StringIO()):
        correct.main([str(json_path), "--speakers", "", *_FORMATS])


def run(preset: str, args: argparse.Namespace) -> int:
    s = load_settings(args.config, args)
    report: list[str] = []
    needs_review = False

    def section(title: str, body: str) -> None:
        report.append(f"## {title}\n\n{body.rstrip()}\n")
        print(f"\n== {title} ==\n{body.rstrip()}")

    # 1. transcribe
    if preset == "check":
        json_path = args.transcript.with_suffix(".json")
        if not json_path.exists():
            print(f"error: {json_path} not found", file=sys.stderr)
            return EXIT_FAILED
    else:
        from video_transcribe import cli
        argv = transcribe_argv(preset, args, s)
        print("transcribing: video-transcribe " + " ".join(argv), file=sys.stderr)
        rc = cli.main(argv)
        if rc != 0:
            print(f"error: transcription failed (exit {rc})", file=sys.stderr)
            return EXIT_FAILED
        json_path = Path(args.video).with_suffix(".json")
        section("Transcribe", f"`video-transcribe {' '.join(argv)}`\n\nwrote {json_path.stem}.txt/.srt/.json")

    # 2. glossary pass -- --speakers "" so a stale default map can never rename anyone
    if s.glossary and s.glossary.exists():
        from video_transcribe import correct
        before = json_path.read_text(encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            correct.main([str(json_path), "--glossary", str(s.glossary), "--speakers", "", *_FORMATS])
        changed = before != json_path.read_text(encoding="utf-8")
        section("Glossary", f"applied {s.glossary.name} -- "
                            + ("text changed" if changed else "no known mis-hearings found"))
    else:
        section("Glossary", "skipped -- no glossary configured")

    # 3. QC
    data = json.loads(json_path.read_text(encoding="utf-8"))
    findings = qc.check_transcript(data)
    review_f = [f for f in findings if f.needs_review]
    needs_review |= bool(review_f)
    body = "\n".join(
        f"- {'REVIEW' if f.needs_review else 'info'} "
        f"{qc._fmt_ts(f.start).strip() or '-'} {f.speaker or ''} {f.detail}".rstrip()
        for f in findings) or "clean -- no repetition, language drift, or impossible timing"
    section("Quality check", f"{len(review_f)} needing review, "
                             f"{len(findings) - len(review_f)} informational\n\n{body}")

    # 4. visual-ID review (only exists for group presets)
    visual = json_path.with_name(json_path.stem + ".visual.json")
    if visual.exists():
        from video_transcribe import visual_id
        rc, out = _capture(visual_id._review, visual, json_path)
        needs_review |= rc == 2
        section("Visual-ID review", f"```\n{out.strip()}\n```")

    # 5. LLM review (opt-in)
    notes = ""
    if args.notes:
        notes = Path(args.notes).read_text(encoding="utf-8")
    if args.notes_text:
        notes = (notes + "\n" + args.notes_text).strip()

    if args.llm:
        try:
            result = llm_review(data, notes=notes, findings=findings,
                                glossary=s.glossary, model=s.llm_model)
        except Exception as e:  # report and keep the local results
            section("LLM review", f"FAILED: {e}")
            needs_review = True
        else:
            utt = data.get("utterances", [])

            def at(i):
                if i is None or not 0 <= i < len(utt):
                    return "-"
                m, sec = divmod(int(utt[i]["start"]), 60)
                return f"{m:02d}:{sec:02d}"

            note_lines = "\n".join(
                f"- **{n.status}** [{at(n.utterance_index)}] {n.note} -- {n.evidence}"
                for n in result.notes) or "(no notes given)"
            needs_review |= any(n.status in ("not_found", "partial") for n in result.notes)
            qc_lines = "\n".join(f"- {v.verdict}: {v.finding} -- {v.comment}"
                                 for v in result.qc_verdicts) or "(none)"
            fix_lines = [f"[{at(f.utterance_index)}] {f.from_text!r} -> {f.to_text!r} "
                         f"({f.confidence}): {f.reason}" for f in result.corrections]

            if args.apply_llm_fixes and result.corrections:
                log = apply_fixes(data, result.corrections)
                json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                _render(json_path)
                fix_body = "\n".join(f"- {line}" for line in log)
            else:
                fix_body = ("\n".join(f"- proposed {line}" for line in fix_lines)
                            + ("\n\nNot applied. Re-run with --apply-llm-fixes to apply the "
                               "high-confidence ones." if fix_lines else "")) or "none proposed"
            section("LLM review", f"{result.summary}\n\n### Notes\n{note_lines}\n\n"
                                  f"### Proposed corrections\n{fix_body}\n\n### QC verdicts\n{qc_lines}")
    elif notes:
        section("Notes", "Notes given but --llm not set, so they were not checked against "
                         "the transcript.\n\n" + notes)

    report_path = json_path.with_name(json_path.stem + ".report.md")
    status = "NEEDS REVIEW" if needs_review else "clean"
    report_path.write_text(f"# {json_path.stem}\n\nStatus: **{status}**\n\n" + "\n".join(report),
                           encoding="utf-8")
    print(f"\nwrote {report_path}\nstatus: {status}")
    return EXIT_REVIEW if needs_review else EXIT_CLEAN


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--notes", type=Path, default=None, help="Your meeting notes (text file).")
    p.add_argument("--notes-text", default=None, help="Your meeting notes, inline.")
    p.add_argument("--llm", action="store_true",
                   help="Also run the Claude review: notes coverage, mis-hearings, QC verdicts. "
                        "Sends the transcript and notes to Anthropic.")
    p.add_argument("--apply-llm-fixes", action="store_true",
                   help="Apply the review's high-confidence corrections, each scoped to the "
                        "one utterance it came from. Default: report them only.")
    p.add_argument("--llm-model", default=None, help=f"Default: {DEFAULT_LLM_MODEL}.")
    p.add_argument("--config", type=Path, default=None, help=f"Default: {CONFIG_PATH}.")
    p.add_argument("--glossary", type=Path, default=None, help="Override the config glossary.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="video-transcribe-run",
                                description="Transcribe, clean, check and optionally "
                                            "LLM-review a meeting in one command.")
    sub = p.add_subparsers(dest="preset", required=True)

    one = sub.add_parser("1on1", help="screen recording + your own mic")
    one.add_argument("video", type=Path)
    one.add_argument("mic", type=Path)
    one.add_argument("--them", required=True, help="The other person's name.")
    one.add_argument("--me", default=None, help="Your name (default from config).")
    one.add_argument("--no-mux", action="store_true", help="Skip writing the combined .mkv.")
    _add_common(one)

    hyb = sub.add_parser("group-hybrid", help="group call video (diarized) + your own mic")
    hyb.add_argument("video", type=Path)
    hyb.add_argument("mic", type=Path)
    hyb.add_argument("--speakers", type=int, default=None, help="Exact speaker count on the video.")
    hyb.add_argument("--me", default=None)
    hyb.add_argument("--roster", default=None, help="Names for visual ID (default from config).")
    hyb.add_argument("--voiceprints", type=Path, default=None)
    hyb.add_argument("--no-mux", action="store_true")
    _add_common(hyb)

    meet = sub.add_parser("group-meet", help="one recording, several people")
    meet.add_argument("video", type=Path)
    meet.add_argument("--speakers", type=int, default=None)
    meet.add_argument("--roster", default=None)
    meet.add_argument("--voiceprints", type=Path, default=None)
    _add_common(meet)

    chk = sub.add_parser("check", help="post-transcription steps on an existing transcript")
    chk.add_argument("transcript", type=Path)
    _add_common(chk)

    args = p.parse_args(argv)
    if args.apply_llm_fixes and not args.llm:
        p.error("--apply-llm-fixes needs --llm")
    return run(args.preset, args)


if __name__ == "__main__":
    raise SystemExit(main())
