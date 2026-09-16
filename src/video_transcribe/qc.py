"""Quality check a finished transcript: hallucination and segmentation defects.

Whisper fails in specific, recognisable ways on real meeting audio. This module
encodes the checks that were previously re-typed by hand after every run, plus
the rules worked out (2026-08/09) for telling a genuine hallucination apart from
a harmless timestamp glitch. Pure stdlib -- no torch/ffmpeg -- so it runs in a
bare environment and in tests.

Checks, and the real defect each one came from:

``non_ascii``
    Whole stretches decoded into another language. Seen once as ~40s of
    Portuguese-looking text in the middle of an English 1-1.

``exact_repeat``
    The same short phrase repeated verbatim many times in a row ("and then
    I'll call it done" x17). Largely suppressed by ``no_repeat_ngram_size=3``
    in transcribe.py, so a hit here now usually means that guard regressed.

``impossible_rate``
    More words than a human mouth can produce in the segment's own duration.
    On its own this is NOT proof of a hallucination -- a segment carrying real,
    coherent speech can get a badly compressed timestamp. What distinguishes a
    real hallucination is impossible rate PLUS repetition: either inside the
    segment or reusing a phrase verbatim from a neighbour. Those are reported
    as ``hallucination``; rate-only hits are reported as ``timing`` and are
    informational. A short interjection ("Yeah.", "Cool.") landing on a 0.02s
    boundary is a rounding artifact, so segments under ``min_words`` are
    ignored entirely.

``long_segment``
    VAD never found a pause and merged minutes of audio into one segment (up
    to 217s observed). Not a defect by itself -- an open mic with room tone
    does this -- but it is the condition the decode loops appear in, so it is
    worth seeing next to the other findings.

``duplicate_speaker``
    The same name listed twice in the transcript's ``speakers``, which inflates
    the speaker count in the header. Happens when two raw diarization labels
    resolve to one person.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = (
    "Finding",
    "check_transcript",
    "find_phrase_repeats",
    "main",
)

# A hallucinated repeat has to be long enough to not be ordinary speech: "yeah
# yeah yeah" is a real thing people say, 17 copies of a six-word clause is not.
MIN_REPEATS = 3
MIN_REPEAT_WORDS = 6
MAX_NGRAM = 10

# Sustained conversational speech tops out around 5 words/sec; 15 leaves room
# for a fast burst before calling the timestamp impossible.
MAX_WORDS_PER_SEC = 15.0
# Past this the segment has essentially no duration to hold its own text (23
# words in 0.04s = 575 w/s was observed). Mild compression -- 15-30 w/s on
# coherent, non-repeating speech -- is a timestamp estimate being wrong; this
# is the timestamp being degenerate, and in every observed case it sat inside
# a hallucinated passage. Flagged for review on rate alone, with wording that
# claims only what is established: the timing is unusable, which makes the
# text unverifiable against the audio.
EXTREME_WORDS_PER_SEC = 60.0
# Below this, a "too fast" reading is timestamp rounding on a clipped
# interjection, not a decode defect. ("Yeah." at 0.06s = 16 w/s.)
MIN_RATE_WORDS = 5

# Phrase length used when testing whether a suspect segment is echoing its
# neighbour. Four words is long enough that a match is not a coincidence of
# common filler ("you know", "I think").
ECHO_NGRAM = 4

# VAD merging this much audio into one segment is worth surfacing.
LONG_SEGMENT_SECONDS = 60.0

_SEVERITY_ORDER = {"hallucination": 0, "duplicate_speaker": 1, "timing": 2, "long_segment": 3}


@dataclass(frozen=True)
class Finding:
    kind: str           # non_ascii | exact_repeat | impossible_rate | long_segment | duplicate_speaker
    severity: str       # hallucination | duplicate_speaker | timing | long_segment
    detail: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None

    @property
    def needs_review(self) -> bool:
        """True for findings that indicate corrupted text, not just odd timing."""
        return self.severity in ("hallucination", "duplicate_speaker")


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def find_phrase_repeats(
    text: str, *, max_n: int = MAX_NGRAM, min_reps: int = MIN_REPEATS,
    min_words: int = MIN_REPEAT_WORDS,
) -> list[tuple[int, int, str]]:
    """Consecutive verbatim repeats as ``(repeat_count, phrase_len, phrase)``.

    Scans for the shortest phrase that repeats back-to-back at each position,
    then skips past the whole run so one loop is reported once. ``min_words``
    is measured over the entire run (count * length), which keeps natural
    short repeats ("yeah yeah yeah") out while catching both a long clause
    repeated 3x and a single word repeated 28x.
    """
    words = _words(text)
    out: list[tuple[int, int, str]] = []
    i = 0
    n_words = len(words)
    while i < n_words:
        hit = False
        for n in range(1, max_n + 1):
            if i + n > n_words:
                continue
            phrase = tuple(words[i:i + n])
            reps = 1
            j = i + n
            while j + n <= n_words and tuple(words[j:j + n]) == phrase:
                reps += 1
                j += n
            if reps >= min_reps and n * reps >= min_words:
                out.append((reps, n, " ".join(phrase)))
                i = j
                hit = True
                break
        if not hit:
            i += 1
    return out


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = _words(text)
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _echoes_neighbour(text: str, neighbours: list[str]) -> str | None:
    """A phrase this segment reuses verbatim from an adjacent segment, if any."""
    mine = _ngrams(text, ECHO_NGRAM)
    for other in neighbours:
        shared = mine & _ngrams(other, ECHO_NGRAM)
        if shared:
            return " ".join(sorted(shared)[0])
    return None


def _self_repeats(text: str) -> str | None:
    """A phrase this segment repeats within itself, if any.

    Uses a lower bar than :func:`find_phrase_repeats` because a single
    hallucinated segment is short: two occurrences of a 3-word phrase anywhere
    in it (not necessarily adjacent) is already the signature.
    """
    words = _words(text)
    seen: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - 2):
        gram = tuple(words[i:i + 3])
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] >= 2:
            return " ".join(gram)
    return None


def check_transcript(
    data: dict, *, max_words_per_sec: float = MAX_WORDS_PER_SEC,
    long_segment: float = LONG_SEGMENT_SECONDS, min_reps: int = MIN_REPEATS,
) -> list[Finding]:
    """Run every check over one parsed transcript ``.json``.

    Accepts the shape written by formats.to_json: ``segments`` (with start/end/
    text/speaker), ``utterances``, and ``speakers``.
    """
    findings: list[Finding] = []
    segments = data.get("segments") or []
    utterances = data.get("utterances") or []

    speakers = data.get("speakers") or []
    dupes = {s for s in speakers if speakers.count(s) > 1}
    for name in sorted(dupes):
        findings.append(Finding(
            kind="duplicate_speaker", severity="duplicate_speaker",
            detail=f"{name!r} listed {speakers.count(name)}x in speakers "
                   f"(inflates the header count)",
            speaker=name,
        ))

    # Text-level checks run over the utterance stream (what a human reads).
    full_text = "\n".join(u.get("text", "") for u in utterances) or \
        "\n".join(s.get("text", "") for s in segments)
    for lineno, line in enumerate(full_text.splitlines(), 1):
        if any(ord(ch) > 127 for ch in line):
            snippet = line.strip()[:80]
            findings.append(Finding(
                kind="non_ascii", severity="hallucination",
                detail=f"line {lineno}: non-ASCII text (language drift?): {snippet!r}",
            ))
    for reps, n, phrase in find_phrase_repeats(full_text, min_reps=min_reps):
        # A bare 3x repeat of a one- or two-word phrase is ordinary speech
        # disfluency ("plug in, plug in, plug in"). A decode loop shows up as
        # either many more repeats or a whole clause coming back verbatim, so
        # only those escalate; the stutters stay visible as info instead of
        # burying the real hits.
        strong = reps >= 4 or n >= 3 or reps * n >= 12
        findings.append(Finding(
            kind="exact_repeat",
            severity="hallucination" if strong else "timing",
            detail=f"{reps}x verbatim repeat of {n}-word phrase: {phrase!r}"
                   + ("" if strong else " (short run -- likely a spoken stutter)"),
        ))

    # Segment-level checks need timestamps.
    texts = [s.get("text", "") for s in segments]
    for i, seg in enumerate(segments):
        start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        duration = end - start
        text = seg.get("text", "")
        n_words = len(text.split())
        speaker = seg.get("speaker")

        if duration >= long_segment:
            findings.append(Finding(
                kind="long_segment", severity="long_segment",
                detail=f"{duration:.0f}s single segment -- VAD found no pause "
                       f"(decode-loop risk zone)",
                start=start, end=end, speaker=speaker,
            ))

        if duration <= 0 or n_words < MIN_RATE_WORDS:
            continue
        rate = n_words / duration
        if rate <= max_words_per_sec:
            continue
        neighbours = [t for t in (texts[i - 1] if i else "", texts[i + 1] if i + 1 < len(texts) else "") if t]
        echo = _echoes_neighbour(text, neighbours)
        internal = _self_repeats(text)
        if echo or internal:
            why = f"echoes neighbour: {echo!r}" if echo else f"repeats itself: {internal!r}"
            findings.append(Finding(
                kind="impossible_rate", severity="hallucination",
                detail=f"{n_words}w in {duration:.2f}s ({rate:.0f} w/s) AND {why}",
                start=start, end=end, speaker=speaker,
            ))
        elif rate >= EXTREME_WORDS_PER_SEC:
            findings.append(Finding(
                kind="impossible_rate", severity="hallucination",
                detail=f"{n_words}w in {duration:.2f}s ({rate:.0f} w/s) -- segment has "
                       f"no duration to hold this text; timing unusable, text "
                       f"unverifiable against audio",
                start=start, end=end, speaker=speaker,
            ))
        else:
            findings.append(Finding(
                kind="impossible_rate", severity="timing",
                detail=f"{n_words}w in {duration:.2f}s ({rate:.0f} w/s), no repetition "
                       f"-- likely a timestamp glitch on real speech",
                start=start, end=end, speaker=speaker,
            ))

    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9),
                                 f.start if f.start is not None else -1.0))
    return findings


def _fmt_ts(seconds: float | None) -> str:
    if seconds is None:
        return "      "
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _report(path: Path, findings: list[Finding], *, quiet: bool) -> int:
    review = [f for f in findings if f.needs_review]
    info = [f for f in findings if not f.needs_review]
    if quiet and not review:
        return 0
    print(f"\n{path.name}")
    if not findings:
        print("  clean -- no repetition, language drift, or impossible timing found")
        return 0
    for f in findings:
        mark = "REVIEW" if f.needs_review else "info  "
        where = f"{_fmt_ts(f.start)}" if f.start is not None else "      "
        who = f" {f.speaker}:" if f.speaker else ""
        print(f"  [{mark}] {where}{who} {f.detail}")
    print(f"  -- {len(review)} needing review, {len(info)} informational")
    return len(review)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="video-transcribe-qc",
        description="Quality check finished transcripts for Whisper hallucination "
                    "and segmentation defects (repetition loops, language drift, "
                    "impossible timestamps, unsplit VAD segments).",
    )
    p.add_argument("transcripts", nargs="*", type=Path,
                   help="transcript .json file(s) produced by video-transcribe")
    p.add_argument("--glob", default=None, metavar="PATTERN",
                   help="Also scan every .json matching this pattern, e.g. "
                        "\"C:\\clips\\*.json\" -- for sweeping a whole folder of "
                        "past transcripts at once.")
    p.add_argument("--max-words-per-sec", type=float, default=MAX_WORDS_PER_SEC,
                   help=f"Flag segments faster than this (default: {MAX_WORDS_PER_SEC}).")
    p.add_argument("--long-segment", type=float, default=LONG_SEGMENT_SECONDS,
                   help=f"Report segments longer than this many seconds "
                        f"(default: {LONG_SEGMENT_SECONDS}).")
    p.add_argument("--min-repeats", type=int, default=MIN_REPEATS,
                   help=f"Consecutive repeats before flagging (default: {MIN_REPEATS}).")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="Emit findings as JSON instead of a report.")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Only print files that have findings needing review.")
    args = p.parse_args(argv)

    paths = list(args.transcripts)
    if args.glob:
        paths += [Path(p) for p in sorted(globlib.glob(args.glob))]
    # A transcript's .txt sits beside its .json; accept either spelling.
    paths = [p.with_suffix(".json") if p.suffix.lower() in (".txt", ".srt", ".vtt") else p
             for p in paths]
    seen: set[Path] = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]
    if not paths:
        print("error: no transcripts given (pass paths and/or --glob)", file=sys.stderr)
        return 2

    total_review = 0
    results: dict[str, list[dict]] = {}
    for path in paths:
        if not path.exists():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"error: {path} is not valid JSON: {e}", file=sys.stderr)
            return 2
        findings = check_transcript(
            data, max_words_per_sec=args.max_words_per_sec,
            long_segment=args.long_segment, min_reps=args.min_repeats,
        )
        if args.as_json:
            results[str(path)] = [asdict(f) for f in findings]
        else:
            total_review += _report(path, findings, quiet=args.quiet)

    if args.as_json:
        print(json.dumps(results, indent=2))
        total_review = sum(1 for fs in results.values() for f in fs
                           if f["severity"] in ("hallucination", "duplicate_speaker"))
    elif len(paths) > 1:
        print(f"\n{len(paths)} transcripts scanned, {total_review} findings need review")

    return 1 if total_review else 0


if __name__ == "__main__":
    raise SystemExit(main())
