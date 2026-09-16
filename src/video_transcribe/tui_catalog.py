"""Task catalog: the single source of truth for every command the TUI exposes.

Each :class:`Task` describes one runnable command -- how to invoke it
(``argv_prefix``, always ``python -m <module> [subcommand]``) and the arguments
it accepts (:class:`Arg`). The TUI renders a widget per arg, and ``build_tokens``
turns the collected form values back into a subprocess argv. This module is
deliberately free of any Textual / torch import so it can be unit-tested and
imported in a bare environment.

Tasks are ordered within each category by how often they're run, and categories
are ordered so the everyday transcription jobs come first.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field, replace
from typing import Literal

from video_transcribe.diarize import DEFAULT_DIARIZE_MODEL
from video_transcribe.visual_id import DEFAULT_NAME_THRESHOLD, DEFAULT_SAMPLE_SPACING
from video_transcribe.voiceprint import DEFAULT_MATCH_THRESHOLD

__all__ = (
    "Arg",
    "Example",
    "Task",
    "CATALOG",
    "CATEGORY_ORDER",
    "CATEGORY_LABELS",
    "CATEGORY_COLORS",
    "ValidationError",
    "build_tokens",
    "grouped_catalog",
    "split_paths",
)

ArgKind = Literal["str", "path", "bool", "choice", "int", "float", "paths"]


class ValidationError(ValueError):
    """A form value can't be turned into a valid argv (missing required field,
    non-numeric integer, out-of-range choice). Surfaced in the TUI before any
    subprocess is spawned, so the user fixes it here instead of reading a
    cryptic argparse error in the log pane."""


@dataclass(frozen=True, slots=True)
class Arg:
    """One configurable option. ``flag=None`` marks a positional argument.

    ``name`` identifies the widget (and must be unique within a task); ``flag``
    is the actual CLI token, kept separate so the on-screen field name and the
    real flag can differ. Defaults are stored as strings for value args so the
    "skip when unchanged" logic in :func:`build_tokens` is a plain string
    compare.
    """

    name: str
    kind: ArgKind
    help: str
    flag: str | None = None
    default: str | bool = ""
    choices: tuple[str, ...] = ()
    required: bool = False
    repeatable: bool = False
    placeholder: str = ""
    # For a bool arg, emit ``[flag, value_when_true]`` instead of a bare flag.
    # Exists for ``--speakers ""``, where the *empty string* is the meaningful
    # value (it suppresses a glossary's default speaker map) and so cannot be
    # typed into a text field -- a blank text field means "unset".
    value_when_true: str | None = None
    # Names of other args in the same task that must not be set at the same
    # time as this one, checked in :func:`build_tokens`.
    conflicts_with: tuple[str, ...] = ()

    @property
    def positional(self) -> bool:
        return self.flag is None


@dataclass(frozen=True, slots=True)
class Example:
    """A ready-made run for a task: a one-line note plus a partial map of
    ``arg name -> value`` (unset args fall back to their defaults). The TUI shows
    the note + the exact command it builds, and can load the values into the
    form. ``values`` must reference only real arg names of the owning task -- the
    ``tui_check`` suite enforces that and that every example builds cleanly."""

    note: str
    values: dict[str, str | bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Task:
    key: str
    label: str
    category: str
    summary: str
    argv_prefix: tuple[str, ...]
    args: tuple[Arg, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)
    examples: tuple[Example, ...] = ()
    # If true, this task sends data to an external LLM (Anthropic Claude).
    llm: bool = False


# --------------------------------------------------------------------------- #
# argv construction
# --------------------------------------------------------------------------- #


def split_paths(text: str) -> list[str]:
    """Split a multi-path field into individual paths.

    One path per line is the primary contract (a Textual ``TextArea`` -- so
    Windows paths with spaces need no quoting). A single line with several
    quoted paths is also honoured via ``shlex`` for people who paste a CLI-style
    list. Surrounding quotes are stripped either way.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1 and ('"' in lines[0] or "'" in lines[0]):
        # posix=False keeps Windows backslashes intact while still honouring
        # double-quoted tokens; strip any residual surrounding quotes.
        return [tok.strip("\"'") for tok in shlex.split(lines[0], posix=False)]
    return [ln.strip("\"'") for ln in lines]


def _check_number(arg: Arg, value: str) -> None:
    caster = int if arg.kind == "int" else float
    try:
        caster(value)
    except ValueError as e:
        kind_word = "an integer" if arg.kind == "int" else "a number"
        raise ValidationError(f"{arg.name}: expected {kind_word}, got {value!r}") from e


def _check_choices(arg: Arg, value: str) -> None:
    if arg.choices and value not in arg.choices:
        raise ValidationError(
            f"{arg.name}: {value!r} is not one of {', '.join(arg.choices)}"
        )


def _positional_tokens(arg: Arg, raw: str) -> list[str]:
    if arg.kind == "paths":
        paths = split_paths(raw)
        if arg.required and not paths:
            raise ValidationError(f"{arg.name}: at least one file is required")
        return paths
    value = raw.strip()
    if not value:
        if arg.required:
            raise ValidationError(f"{arg.name}: required")
        return []
    return [value]


def _flag_tokens(arg: Arg, value: str | bool) -> list[str]:
    assert arg.flag is not None  # positionals routed elsewhere
    if arg.kind == "bool":
        if not value:
            return []
        return [arg.flag] if arg.value_when_true is None else [arg.flag, arg.value_when_true]
    text = str(value).strip()
    if not text:
        if arg.required:
            raise ValidationError(f"{arg.name}: required")
        return []
    # A value equal to the tool's own default adds nothing but noise to the
    # command; drop it so the preview shows only what the user actually changed.
    if not arg.required and not arg.repeatable and text == str(arg.default).strip():
        return []
    if arg.repeatable:
        out: list[str] = []
        for item in (p.strip() for p in text.split(",")):
            if not item:
                continue
            _check_choices(arg, item)
            out += [arg.flag, item]
        return out
    if arg.kind in ("int", "float"):
        _check_number(arg, text)
    _check_choices(arg, text)
    return [arg.flag, text]


def build_tokens(task: Task, values: dict[str, str | bool]) -> list[str]:
    """Turn collected form values into argv tokens (everything after ``python``).

    Flags are emitted first, positionals last -- ``argparse`` accepts optionals
    before a trailing ``nargs="+"`` positional, so this ordering is unambiguous
    for the ``video-transcribe FILES...`` shape and for the subcommand shapes
    alike. Raises :class:`ValidationError` on the first bad field.
    """
    def _is_set(name: str) -> bool:
        arg = next((a for a in task.args if a.name == name), None)
        if arg is None:
            return False
        raw = values.get(name, arg.default)
        return bool(raw) if arg.kind == "bool" else bool(str(raw).strip())

    flags: list[str] = []
    positionals: list[str] = []
    for arg in task.args:
        raw = values.get(arg.name, arg.default)
        if arg.conflicts_with and _is_set(arg.name):
            clash = [n for n in arg.conflicts_with if _is_set(n)]
            if clash:
                raise ValidationError(
                    f"{arg.name}: can't be combined with {', '.join(clash)} "
                    f"-- they set the same flag"
                )
        if arg.positional:
            positionals += _positional_tokens(arg, str(raw))
        else:
            flags += _flag_tokens(arg, raw)
    return [*task.argv_prefix, *flags, *positionals]


def grouped_catalog() -> dict[str, list[Task]]:
    """Catalog grouped by category, in ``CATEGORY_ORDER`` then insertion order."""
    grouped: dict[str, list[Task]] = {cat: [] for cat in CATEGORY_ORDER}
    for task in CATALOG.values():
        grouped.setdefault(task.category, []).append(task)
    return {cat: tasks for cat, tasks in grouped.items() if tasks}


# --------------------------------------------------------------------------- #
# shared argument definitions
# --------------------------------------------------------------------------- #

_FMT_CHOICES = ("txt", "srt", "vtt", "json")

INPUTS = Arg("inputs", "paths", "Video/audio file(s) -- one per line.", required=True,
             placeholder="C:\\clips\\talk.mp4")
FORMAT = Arg("format", "str", "Output formats, comma-separated.", flag="--format",
             choices=_FMT_CHOICES, repeatable=True, placeholder="txt,srt,vtt,json")
MODEL = Arg("model", "str", "Whisper model (large-v3 = highest quality).", flag="--model",
            default="large-v3", choices=("large-v3", "large-v3-turbo", "base",
                                         "large-v3-turbo-fp16"), placeholder="large-v3 | large-v3-turbo")
LANGUAGE = Arg("language", "str", "Language code (blank = auto-detect).", flag="--language",
               placeholder="en, ko, hi, ...")
DEVICE = Arg("device", "choice", "Inference device (cuda = NVIDIA GPU only).", flag="--device",
             default="cpu", choices=("cpu", "cuda", "auto"))
COMPUTE = Arg("compute_type", "str", "CTranslate2 compute type.", flag="--compute-type",
              default="int8", placeholder="int8 | float16")
HOTWORDS = Arg("hotwords", "str", "Bias terms (names/jargon), comma-separated.", flag="--hotwords",
               placeholder="GrimeReaper, pyannote")
HOTWORDS_FILE = Arg("hotwords_file", "path", "Hotwords file (.json or one term per line).",
                    flag="--hotwords-file")
OUTPUT_DIR = Arg("output_dir", "path", "Output directory (blank = beside input file).",
                 flag="--output-dir")
SPEAKER = Arg("speaker", "str", "Label the whole transcript with one speaker name.",
              flag="--speaker")
NO_VAD = Arg("no_vad", "bool", "Disable voice-activity-detection (captures more, slower).",
             flag="--no-vad")
NO_TIDY = Arg("no_tidy", "bool", "Skip the light readability pass (keep raw casing/spacing).",
              flag="--no-tidy")
NO_PUNCT = Arg("no_punctuate", "bool", "Skip ML punctuation/sentence restoration.",
               flag="--no-punctuate")
KEEP_AUDIO = Arg("keep_audio", "bool", "Keep the intermediate 16 kHz WAV file.",
                 flag="--keep-audio")
VERBOSE = Arg("verbose", "bool", "Stream each segment as it is decoded.", flag="--verbose")
QUIET = Arg("quiet", "bool", "Suppress progress output.", flag="--quiet")

HF_TOKEN = Arg("hf_token", "str", "Hugging Face token (else uses $HF_TOKEN env var).",
               flag="--hf-token")
SPEAKERS = Arg("speakers", "int", "Exact number of speakers, if known (more accurate).",
               flag="--speakers")
MIN_SPK = Arg("min_speakers", "int", "Lower bound on number of speakers.",
              flag="--min-speakers")
MAX_SPK = Arg("max_speakers", "int", "Upper bound on number of speakers.",
              flag="--max-speakers")
DIARIZE_MODEL = Arg("diarize_model", "str", "pyannote pipeline model.",
                    flag="--diarize-model", default=DEFAULT_DIARIZE_MODEL)
VOICEPRINTS = Arg("voiceprints", "path", "Voiceprint store JSON (auto-name speakers by voice).",
                  flag="--voiceprints")
VOICE_THRESHOLD = Arg("voice_threshold", "float", "Cosine-similarity threshold for a voice match.",
                      flag="--voice-threshold", default=str(DEFAULT_MATCH_THRESHOLD))

# Visual ID args
VISUAL_ID = Arg("visual_id", "bool",
                "OCR active-speaker tiles in Google Meet gallery-view to auto-name diarized speakers.",
                flag="--visual-id")
VISUAL_ROSTER = Arg("visual_roster", "str",
                    "Known participant names (comma-separated) for visual matching.",
                    flag="--visual-roster", placeholder="Mar, Sharad, John")
VISUAL_REPORT = Arg("visual_report", "path",
                    "Write the visual-id detection report JSON here.", flag="--visual-report")
VISUAL_SAMPLE_SPACING = Arg("visual_sample_spacing", "float",
                            "Frame-sampling interval (seconds) for visual ID.",
                            flag="--visual-sample-spacing", default=str(DEFAULT_SAMPLE_SPACING))
VISUAL_NAME_THRESHOLD = Arg("visual_name_threshold", "float",
                            "Confidence threshold for a visual name match.",
                            flag="--visual-name-threshold", default=str(DEFAULT_NAME_THRESHOLD))


# The shared trailing block reused by every transcription mode, in a sensible
# tab order (output shape first, then quality knobs, then flags).
_COMMON_TAIL = (FORMAT, MODEL, LANGUAGE, DEVICE, COMPUTE, HOTWORDS, HOTWORDS_FILE,
                OUTPUT_DIR, NO_VAD, NO_TIDY, NO_PUNCT, KEEP_AUDIO, VERBOSE, QUIET)

# Shorter tail for the track-based modes, which don't expose device/compute.
# HOTWORDS_FILE belongs here as much as in _COMMON_TAIL: in practice every
# real multi-track run passes a glossary file to bias names, and leaving the
# flag out of the form meant retyping the whole command by hand.
_TRACK_TAIL = (FORMAT, MODEL, LANGUAGE, HOTWORDS, HOTWORDS_FILE, OUTPUT_DIR,
               NO_TIDY, NO_PUNCT, VERBOSE, QUIET)

MUX = Arg("mux", "bool", "Also write a combined .mkv (Mix + Desktop + Mic).", flag="--mux")
TRACK_SPEAKERS = Arg("track_speakers", "str",
                     "Speaker names, one per file, in the same order as the files.",
                     flag="--track-speakers", required=True, placeholder="Mar,Sharad")


# --------------------------------------------------------------------------- #
# the catalog
# --------------------------------------------------------------------------- #

CATEGORY_ORDER = (
    "transcribe",   # everyday transcription jobs
    "verify",       # checking a finished transcript before trusting it
    "speakers",     # speaker diarization & voiceprints
    "media",        # media manipulation (mux)
    "visual",       # visual speaker identification
    "correct",      # post-processing & correction
    "setup",        # environment checks & setup
)

CATEGORY_LABELS = {
    "transcribe": "Transcribe",
    "verify": "Check Transcript",
    "speakers": "Speaker Diarization",
    "media": "Media",
    "visual": "Visual ID",
    "correct": "Correct & Clean",
    "setup": "Setup & Checks",
}

# Category sidebar colors for quick visual scanning.
CATEGORY_COLORS: dict[str, str] = {
    "transcribe": "#7AE582",
    "verify": "#FFB347",
    "speakers": "#E09BFF",
    "media": "#7AE0E5",
    "visual": "#FFD862",
    "correct": "#FF8A8A",
    "setup": "#A0A0A0",
}

_TASKS: tuple[Task, ...] = (
    # ── Transcribe ──────────────────────────────────────────────────────
    # Ordered by how often each one is actually run. The 1-1 sync at the top
    # accounts for more runs than everything below it combined.

    Task(
        key="sync-1on1",
        label="1-1 sync (video + own mic) — the usual run",
        category="transcribe",
        summary="ReLive video + your own mic file, merged by timestamp into one "
                "exact two-speaker transcript. Pre-set to the combination every "
                "biweekly sync uses: txt+srt+json, --mux, glossary hotwords, "
                "streaming output. VIDEO first, MIC second — --track-speakers "
                "names them in that order.",
        argv_prefix=("-m", "video_transcribe"),
        args=(
            INPUTS, TRACK_SPEAKERS,
            replace(MUX, default=True),
            replace(FORMAT, default="txt,srt,json"),
            HOTWORDS_FILE,
            replace(VERBOSE, default=True),
            MODEL, LANGUAGE, HOTWORDS, OUTPUT_DIR, NO_TIDY, NO_PUNCT, QUIET,
        ),
        tags=("transcribe", "tracks", "mux", "1-1", "sync", "relive", "everyday"),
    ),
    Task(
        key="transcribe",
        label="Transcribe — basic",
        category="transcribe",
        summary="Transcribe one or more video/audio files using Whisper "
                "(large-v3 by default). Outputs a readable .txt transcript "
                "beside each input. Add --format for .srt/.vtt/.json. "
                "Use --speaker to tag a single-presenter recording.",
        argv_prefix=("-m", "video_transcribe"),
        args=(INPUTS, SPEAKER, *_COMMON_TAIL),
        tags=("transcribe", "whisper", "basic"),
    ),
    Task(
        key="transcribe-fast",
        label="Transcribe — fast (turbo)",
        category="transcribe",
        summary="Transcribe using large-v3-turbo for ~2x speed at slightly "
                "lower quality. Good for quick drafts or when you just need "
                "the gist. Use --no-vad to capture every utterance.",
        argv_prefix=("-m", "video_transcribe"),
        args=(INPUTS, SPEAKER, *_COMMON_TAIL),
        tags=("transcribe", "whisper", "fast", "turbo"),
    ),
    Task(
        key="transcribe-all-formats",
        label="Transcribe — all output formats",
        category="transcribe",
        summary="Transcribe and write all four output formats: .txt, .srt, "
                ".vtt, and .json. Useful when you need subtitles and the "
                "structured data for further processing.",
        argv_prefix=("-m", "video_transcribe"),
        args=(INPUTS, SPEAKER, *_COMMON_TAIL),
        tags=("transcribe", "whisper", "all-formats"),
    ),
    Task(
        key="transcribe-english",
        label="Transcribe — force English",
        category="transcribe",
        summary="Transcribe with language forced to English, skipping the "
                "auto-detect step for slightly faster processing. "
                "Use when the recording is definitely in English.",
        argv_prefix=("-m", "video_transcribe"),
        args=(INPUTS, *_COMMON_TAIL),
        tags=("transcribe", "english", "whisper"),
    ),

    # ── Check Transcript ────────────────────────────────────────────────
    # Run after almost every transcription -- second only to the 1-1 sync
    # itself. Whisper's failure modes on real meeting audio are quiet: the
    # text stays fluent and plausible while repeating or drifting, so a
    # transcript that reads fine can still be wrong.

    Task(
        key="qc",
        label="Check transcript for hallucinations",
        category="verify",
        summary="Scan a finished transcript .json for Whisper's known failure "
                "modes: verbatim repetition loops, non-English drift, segments "
                "holding more words than their own duration allows, unsplit VAD "
                "segments, and duplicate speaker names. Findings are split into "
                "REVIEW (text is probably corrupted) and info (odd timing on "
                "text that is probably fine) -- impossible speed alone is not "
                "treated as proof, it has to come with repetition. Read-only; "
                "exits non-zero when something needs review.",
        argv_prefix=("-m", "video_transcribe.qc"),
        args=(
            Arg("transcripts", "paths", "Transcript .json file(s) -- one per line.",
                placeholder="C:\\clips\\meeting.json"),
            Arg("glob", "str", "Also scan every .json matching this pattern.",
                flag="--glob", placeholder="C:\\clips\\*.json"),
            Arg("max_words_per_sec", "float", "Flag segments faster than this.",
                flag="--max-words-per-sec", default="15.0"),
            Arg("long_segment", "float", "Report segments longer than this (seconds).",
                flag="--long-segment", default="60.0"),
            Arg("min_repeats", "int", "Consecutive repeats before flagging.",
                flag="--min-repeats", default="3"),
            Arg("as_json", "bool", "Emit findings as JSON instead of a report.", flag="--json"),
            Arg("quiet", "bool", "Only print files that have findings needing review.",
                flag="--quiet"),
        ),
        tags=("verify", "qc", "hallucination", "repetition", "check", "quality"),
    ),
    Task(
        key="visual-review",
        label="Review visual-ID report (merge_suspected)",
        category="verify",
        summary="Summarise an existing .visual.json: which diarized labels are "
                "merge_suspected (one label, more than one person), what each "
                "label reads as visually, and which labels stayed generic. Ends "
                "with a ready-to-edit correct.py --speaker-at command that "
                "applies your decision. A flagged label is never split "
                "automatically, so this is the step between detection and fix. "
                "Check the names before running the suggested command -- Meet's "
                "active-speaker highlight lingers, so a short utterance right "
                "after a speaker change can read as the previous person.",
        argv_prefix=("-m", "video_transcribe.visual_id", "review"),
        args=(
            Arg("report", "path", "The .visual.json written by a --visual-id run.",
                required=True),
            Arg("transcript", "path", "Matching transcript .json (shows each flagged "
                                      "utterance's text).", flag="--transcript"),
        ),
        tags=("verify", "visual", "merge", "review", "speakers"),
    ),

    # ── Speaker Diarization ─────────────────────────────────────────────

    Task(
        key="diarize",
        label="Transcribe + diarize speakers",
        category="speakers",
        summary="Transcribe and label who-said-what using pyannote diarization. "
                "Needs a Hugging Face token (--hf-token or $HF_TOKEN) and the "
                "'diarize' extra (uv sync --extra diarize). Pass --speakers if "
                "you know the count for more accuracy.",
        argv_prefix=("-m", "video_transcribe", "--diarize"),
        args=(INPUTS, SPEAKERS, MIN_SPK, MAX_SPK, HF_TOKEN, DIARIZE_MODEL,
              VOICEPRINTS, VOICE_THRESHOLD, *_COMMON_TAIL),
        tags=("transcribe", "diarize", "speakers", "pyannote"),
    ),
    Task(
        key="diarize-known-speakers",
        label="Diarize — known speaker count",
        category="speakers",
        summary="Transcribe with diarization, telling pyannote the exact number "
                "of speakers. This dramatically improves accuracy when you know "
                "the count (e.g. a 2-person interview or 4-person panel).",
        argv_prefix=("-m", "video_transcribe", "--diarize"),
        args=(INPUTS, SPEAKERS, HF_TOKEN, *_COMMON_TAIL),
        tags=("transcribe", "diarize", "speakers", "known-count"),
    ),

    # ── Multi-track ─────────────────────────────────────────────────────

    Task(
        key="list-tracks",
        label="List audio tracks",
        category="transcribe",
        summary="Inspect a file's audio tracks and exit. Print each track's "
                "index, codec, and channels. Run this first to find track "
                "indices for the by-track transcription modes below.",
        argv_prefix=("-m", "video_transcribe", "--list-tracks"),
        args=(INPUTS,),
        tags=("inspect", "tracks", "list"),
    ),
    Task(
        key="tracks-in-file",
        label="Transcribe by track (one file)",
        category="transcribe",
        summary="Multi-track file (e.g. AMD ReLive recording with a separate "
                "mic muxed in): transcribe each track separately and label by "
                "track index. Exact speakers, no diarization needed. Use "
                "--tracks '0=Desktop,1=Mic' to name each track.",
        argv_prefix=("-m", "video_transcribe"),
        args=(
            INPUTS,
            Arg("tracks", "str", "Track → speaker map: '0=Mar,1=Sharad'.", flag="--tracks",
                required=True, placeholder="0=Mar,1=Sharad"),
            *_TRACK_TAIL,
        ),
        tags=("transcribe", "tracks", "speakers", "relive"),
    ),
    Task(
        key="tracks-files",
        label="Transcribe parallel track files",
        category="transcribe",
        summary="Separate files for one recording (e.g. video.mp4 + mic.m4a): "
                "transcribe each and merge by timestamp, labeled by "
                "--track-speakers. Also supports --mux to write a combined .mkv.",
        argv_prefix=("-m", "video_transcribe"),
        args=(INPUTS, TRACK_SPEAKERS, MUX, *_TRACK_TAIL),
        tags=("transcribe", "tracks", "mux", "speakers", "relive"),
    ),
    Task(
        key="hybrid",
        label="Hybrid — diarize + fixed tracks",
        category="speakers",
        summary="Group call + your own mic: acoustically diarize one input "
                "(--diarize-track) while other files are fixed single-speaker "
                "tracks. Best of both worlds for meetings where you have a "
                "separate mic but the video has multiple voices. Layer on "
                "voiceprints to auto-name known voices and --visual-id to catch "
                "a cluster that merged two people -- that combination is what "
                "the team engineering meetings use.",
        argv_prefix=("-m", "video_transcribe"),
        args=(
            INPUTS,
            Arg("diarize_track", "int", "0-based index of the file to diarize.",
                flag="--diarize-track", required=True, placeholder="0"),
            replace(TRACK_SPEAKERS, help="Names for the OTHER (non-diarized) files.",
                    placeholder="Sharad"),
            SPEAKERS, MIN_SPK, MAX_SPK, HF_TOKEN, VOICEPRINTS, VOICE_THRESHOLD,
            VISUAL_ID, VISUAL_ROSTER, VISUAL_REPORT, MUX, *_TRACK_TAIL,
        ),
        tags=("transcribe", "diarize", "tracks", "hybrid", "speakers", "visual"),
    ),

    # ── Media ───────────────────────────────────────────────────────────

    Task(
        key="mux",
        label="Mux video + mic → MKV",
        category="media",
        summary="Combine a video file (with desktop audio) + a separate "
                "microphone file into one .mkv with three audio tracks: a "
                "default Mix track plus isolated Desktop and Mic tracks. "
                "Video is stream-copied (no re-encode).",
        argv_prefix=("-m", "video_transcribe.mux"),
        args=(
            Arg("video", "path", "Video file (with desktop/system audio).", required=True),
            Arg("mic", "path", "Separate microphone audio file.", required=True),
            Arg("output", "path", "Output .mkv (blank = <video>.with-mic.mkv).", flag="--output"),
        ),
        tags=("media", "mux", "ffmpeg"),
    ),

    # ── Visual ID ───────────────────────────────────────────────────────

    Task(
        key="visual-id",
        label="Visual speaker ID (Google Meet)",
        category="visual",
        summary="OCR the active-speaker-highlighted tile's name label in a "
                "Google Meet gallery-view recording to auto-name diarized "
                "speakers, and flag any label whose utterances resolve to more "
                "than one person. Fully local (EasyOCR) — no external API "
                "calls. Only ever fills in still-generic 'Speaker N' labels; a "
                "name already set by voiceprints or a fixed track is never "
                "overwritten, and a merge_suspected label is never split "
                "automatically. Works on gallery-view screen recordings; a "
                "Meet cloud recording in speaker view has no tile borders to "
                "read and will find nothing. Follow up with 'Review visual-ID "
                "report'.",
        argv_prefix=("-m", "video_transcribe", "--diarize", "--visual-id"),
        args=(
            INPUTS,
            SPEAKERS, MIN_SPK, MAX_SPK, HF_TOKEN, DIARIZE_MODEL,
            VISUAL_ROSTER, VISUAL_REPORT, VISUAL_SAMPLE_SPACING,
            VISUAL_NAME_THRESHOLD, VOICEPRINTS, VOICE_THRESHOLD,
            *_COMMON_TAIL,
        ),
        tags=("visual", "meet", "ocr", "speakers", "diarize"),
    ),

    # ── Correct ─────────────────────────────────────────────────────────

    Task(
        key="correct",
        label="Correct — glossary (local, no LLM)",
        category="correct",
        summary="Apply fixes to a finished transcript .json: glossary term "
                "corrections, whole-label speaker renames, per-utterance "
                "speaker overrides, and one-off text fixes -- in any "
                "combination, in one pass. Fully local and deterministic: no "
                "re-transcription, no external API. Every field is optional "
                "except the transcript, so this is also how you simply re-emit "
                "txt/srt/vtt/json from an edited .json.",
        argv_prefix=("-m", "video_transcribe.correct"),
        args=(
            Arg("input", "path", "Transcript .json from video-transcribe.", required=True),
            Arg("glossary", "path", "Glossary JSON (speaker_map + corrections). Optional.",
                flag="--glossary"),
            Arg("speakers", "str", "Whole-label renames: 'Speaker 1=Noah,Speaker 2=Mark'.",
                flag="--speakers", placeholder="Speaker 1=Noah,Speaker 2=Mark",
                conflicts_with=("suppress_speaker_map",)),
            Arg("suppress_speaker_map", "bool",
                "Ignore the glossary's built-in speaker_map entirely (sends --speakers \"\"). "
                "Use when the transcript's names are already correct and a stale default "
                "map meant for a different meeting would overwrite them.",
                flag="--speakers", value_when_true="",
                conflicts_with=("speakers",)),
            Arg("speaker_at", "str", "Per-utterance override by start time (repeatable).",
                flag="--speaker-at", repeatable=True, placeholder="37.3=Sharad"),
            Arg("correction", "str", "One-off text fix FROM=TO (repeatable).",
                flag="--correction", repeatable=True, placeholder="Pral=Carolyn"),
            OUTPUT_DIR,
            Arg("format", "str", "Output formats (default: txt,srt,json).", flag="--format",
                choices=_FMT_CHOICES, repeatable=True, placeholder="txt,srt,json"),
        ),
        tags=("correct", "glossary", "speakers", "local", "post-process"),
    ),
    Task(
        key="correct-speaker-at",
        label="Correct — split a merged speaker label",
        category="correct",
        summary="Reassign ONE utterance to a different speaker by its start "
                "time, leaving every other utterance under that label alone. "
                "This is how a visual-ID merge_suspected finding gets fixed: "
                "diarization put two people in one cluster, and only you can "
                "say which utterance belongs to whom. Get the timestamps from "
                "'Review visual-ID report', and sanity-check them first -- "
                "Meet's speaker highlight lingers, so the visual reading right "
                "after a speaker change can name the previous person.",
        argv_prefix=("-m", "video_transcribe.correct"),
        args=(
            Arg("input", "path", "Transcript .json from video-transcribe.", required=True),
            Arg("speaker_at", "str", "Time=Name overrides (repeatable), e.g. '37.3=Sharad'.",
                flag="--speaker-at", repeatable=True, required=True,
                placeholder="37.3=Sharad"),
            Arg("glossary", "path", "Glossary JSON, if you also want term fixes.",
                flag="--glossary"),
            OUTPUT_DIR,
            Arg("format", "str", "Output formats (default: txt,srt,json).", flag="--format",
                choices=_FMT_CHOICES, repeatable=True, placeholder="txt,srt,json"),
        ),
        tags=("correct", "speaker", "merge", "split", "local"),
    ),
    Task(
        key="llm-correct",
        label="LLM ✦ Correction (Claude API — sends text off-machine)",
        category="correct",
        summary="Correction-only pass over a transcript .json via the Claude "
                "API: fixes misheard names/jargon and obvious ASR slips. "
                "Sends transcript text off-machine to Anthropic; costs a few "
                "cents per meeting. Writes separate .llm.* files + a diff. "
                "Needs the 'llm' extra + ANTHROPIC_API_KEY.",
        argv_prefix=("-m", "video_transcribe.llm_correct"),
        args=(
            Arg("input", "path", "Transcript .json from video-transcribe.", required=True),
            Arg("glossary", "path", "Optional glossary JSON for context.", flag="--glossary"),
            Arg("model", "str", "Claude model name.", flag="--model", default="claude-opus-4-8"),
            OUTPUT_DIR,
            Arg("format", "str", "Output formats (default: txt,json).", flag="--format",
                choices=("txt", "json"), repeatable=True, placeholder="txt,json"),
        ),
        tags=("correct", "llm", "claude", "api", "off-machine"),
        llm=True,
    ),

    # ── Voiceprints ─────────────────────────────────────────────────────

    Task(
        key="voiceprint-list",
        label="Voiceprints: list enrolled",
        category="speakers",
        summary="List all enrolled people in a voiceprint store and how many "
                "voice samples each has. A sanity check before using "
                "voiceprints for auto-naming speakers.",
        argv_prefix=("-m", "video_transcribe.voiceprint", "list"),
        args=(Arg("store", "path", "Voiceprint store JSON.", flag="--store", required=True),),
        tags=("voiceprint", "list", "enrolled"),
    ),
    Task(
        key="voiceprint-enroll",
        label="Voiceprints: enroll speakers",
        category="speakers",
        summary="Grow a voiceprint store from an already-corrected transcript "
                ".json + its source media. Future diarized recordings will then "
                "auto-name these voices. Use --names to enroll only specific people.",
        argv_prefix=("-m", "video_transcribe.voiceprint", "enroll"),
        args=(
            Arg("transcript", "path", "Corrected transcript .json (real speaker names).", required=True),
            Arg("media", "path", "Audio/video file that speech came from.", required=True),
            Arg("store", "path", "Voiceprint store JSON (created/updated).", flag="--store", required=True),
            Arg("names", "str", "Only enroll these speakers (comma-separated; default: all).", flag="--names"),
            Arg("track", "int", "0-based audio track index for multi-track file.", flag="--track"),
            HF_TOKEN,
        ),
        tags=("voiceprint", "enroll", "speaker-identification"),
    ),
    Task(
        key="voiceprint-validate",
        label="Voiceprints: validate against transcript",
        category="speakers",
        summary="Check a store's matches against a transcript whose speaker "
                "names you've already confirmed — a sanity check before "
                "trusting the store for auto-naming on future recordings.",
        argv_prefix=("-m", "video_transcribe.voiceprint", "validate"),
        args=(
            Arg("transcript", "path", "Corrected transcript .json (real speaker names).", required=True),
            Arg("media", "path", "Audio/video file that speech came from.", required=True),
            Arg("store", "path", "Voiceprint store JSON.", flag="--store", required=True),
            Arg("names", "str", "Only validate these speakers (comma-separated).", flag="--names"),
            Arg("track", "int", "0-based audio track index for multi-track file.", flag="--track"),
            Arg("threshold", "float", "Match confidence threshold.", flag="--threshold",
                default=str(DEFAULT_MATCH_THRESHOLD)),
            HF_TOKEN,
        ),
        tags=("voiceprint", "validate", "sanity-check"),
    ),

    # ── Setup ───────────────────────────────────────────────────────────

    Task(
        key="doctor",
        label="Environment doctor",
        category="setup",
        summary="Check the local setup: ffmpeg/ffprobe on PATH, Python version, "
                "and all optional extras (diarize, readable, llm, visual, tui). "
                "Also checks Hugging Face and Anthropic tokens. "
                "Run this first if a job fails to start.",
        argv_prefix=("-m", "video_transcribe.tui_doctor"),
        args=(),
        tags=("setup", "doctor", "check", "diagnostics"),
    ),
)

# Ready-made presets / examples, grounded in this project's actual usage:
# AMD ReLive recordings (video + separate mic), glossary corrections, Claude
# passes, Google Meet recordings, etc. File names are placeholders; edit the
# paths in the form before running.
_EXAMPLES: dict[str, tuple[Example, ...]] = {
    "sync-1on1": (
        Example("Biweekly 1-1: ReLive video + own mic, muxed, glossary hotwords",
                {"inputs": "LAB.Mar.BiweeklySync.Sync14.mp4\nLAB.Mar.BiweeklySync.Sync14.m4a",
                 "track_speakers": "Mar,Sharad",
                 "hotwords_file": "transcript-glossary.json"}),
        Example("Playtest call with a different person on the far end",
                {"inputs": "LAB.LoomCradle.Playtest.JV.mp4\nLAB.LoomCradle.Playtest.JV.m4a",
                 "track_speakers": "JV,Sharad",
                 "hotwords_file": "transcript-glossary.json"}),
        Example("Transcript only -- skip the .mkv (it already exists from an earlier run)",
                {"inputs": "sync.mp4\nsync.m4a", "track_speakers": "Mar,Sharad",
                 "mux": False}),
    ),
    "qc": (
        Example("Check the transcript you just produced",
                {"transcripts": "meeting.json"}),
        Example("Sweep every past transcript in a folder, listing only real problems",
                {"glob": "C:\\Users\\me\\Videos\\*.json", "quiet": True}),
        Example("Machine-readable findings for a script to consume",
                {"transcripts": "meeting.json", "as_json": True}),
    ),
    "visual-review": (
        Example("Read the merge_suspected findings and get the fix command",
                {"report": "meeting.visual.json", "transcript": "meeting.json"}),
    ),
    "transcribe": (
        Example("Highest quality (large-v3); transcript beside the input",
                {"inputs": "talk.mp4"}),
        Example("Also write SRT + VTT subtitles alongside the text",
                {"inputs": "talk.mp4", "format": "txt,srt,vtt"}),
        Example("Single presenter -- label with one name",
                {"inputs": "lecture.mp4", "speaker": "Sharad"}),
        Example("Quick draft with turbo model, also write JSON",
                {"inputs": "clip.mp4", "model": "large-v3-turbo", "format": "txt,json"}),
        Example("Multi-file batch transcription",
                {"inputs": "clip1.mp4\nclip2.mp4\nclip3.mp4"}),
    ),
    "transcribe-fast": (
        Example("Fast turbo transcribe with all output formats",
                {"inputs": "quick-draft.mp4", "model": "large-v3-turbo",
                 "format": "txt,srt,vtt,json"}),
    ),
    "transcribe-all-formats": (
        Example("Transcribe and write every output format",
                {"inputs": "full-meeting.mp4", "format": "txt,srt,vtt,json"}),
    ),
    "transcribe-english": (
        Example("Force English for a known-English recording",
                {"inputs": "english-talk.mp4", "language": "en"}),
    ),
    "diarize": (
        Example("Label speakers (needs HF token); write txt + srt",
                {"inputs": "meeting.mp4", "format": "txt,srt"}),
        Example("Tell pyannote there are exactly 2 speakers",
                {"inputs": "interview.mp4", "speakers": "2"}),
        Example("Auto-name known voices from a voiceprint store",
                {"inputs": "meeting.mp4", "speakers": "4", "voiceprints": "voiceprints.json"}),
    ),
    "diarize-known-speakers": (
        Example("3-person panel discussion",
                {"inputs": "panel.mp4", "speakers": "3"}),
        Example("2-person interview",
                {"inputs": "interview.mp4", "speakers": "2"}),
    ),
    "list-tracks": (
        Example("Inspect a ReLive recording's audio tracks (find the indices)",
                {"inputs": "meeting.mp4"}),
    ),
    "tracks-in-file": (
        Example("ReLive mic muxed into the video: label exactly by track",
                {"inputs": "meeting.mp4", "tracks": "0=Mar,1=Sharad"}),
    ),
    "tracks-files": (
        Example("Video + separate mic file, merged by timestamp",
                {"inputs": "meeting.mp4\nmeeting.m4a", "track_speakers": "Mar,Sharad"}),
        Example("...and also write one combined .mkv (Mix + Desktop + Mic)",
                {"inputs": "meeting.mp4\nmeeting.m4a", "track_speakers": "Mar,Sharad", "mux": True}),
    ),
    "hybrid": (
        Example("Group call (diarize file 0) + your own separate mic",
                {"inputs": "meeting.mp4\nmeeting.m4a", "diarize_track": "0",
                 "speakers": "4", "track_speakers": "Sharad"}),
        Example("Team engineering meeting: diarize + voiceprints + visual ID + mux",
                {"inputs": "LAB.EngineeringMeeting.mp4\nLAB.EngineeringMeeting.m4a",
                 "diarize_track": "0", "track_speakers": "Sharad",
                 "voiceprints": "voiceprints.json", "voice_threshold": "0.75",
                 "visual_id": True,
                 "visual_roster": "Ryan,Mar,Ness,John,Sharad,Carolyn,Fabricio,Jacob,Noah",
                 "mux": True, "hotwords_file": "transcript-glossary.json"}),
    ),
    "mux": (
        Example("Merge video + separate mic into one playable .mkv",
                {"video": "meeting.mp4", "mic": "meeting.m4a"}),
    ),
    "visual-id": (
        Example("Google Meet gallery-view: OCR name labels + diarize",
                {"inputs": "google-meet.mp4", "speakers": "5",
                 "visual_roster": "Mar,Sharad,Ryan,Ness,John"}),
        Example("Team meeting: voiceprints for known voices, visual ID for the rest",
                {"inputs": "LabradorTeamMeeting.mp4",
                 "voiceprints": "voiceprints.json", "voice_threshold": "0.75",
                 "visual_roster": "Ryan,Mar,Ness,John,Sharad,Carolyn,Noah,Jacob",
                 "hotwords_file": "transcript-glossary.json"}),
    ),
    "correct": (
        Example("Apply glossary term fixes + speaker names to a transcript",
                {"input": "meeting.json", "glossary": "g.json"}),
        Example("Name the speakers diarization left generic",
                {"input": "meeting.json",
                 "speakers": "Speaker 1=Noah,Speaker 2=Mark,Speaker 3=Will"}),
        Example("One-off fixes this recording only, no glossary edit",
                {"input": "meeting.json", "correction": "Pral=Carolyn"}),
        Example("Suppress a stale glossary speaker_map, keep its term fixes",
                {"input": "meeting.json", "glossary": "g.json",
                 "suppress_speaker_map": True}),
    ),
    "correct-speaker-at": (
        Example("Split a merged label: the 37.3s utterance was actually Sharad",
                {"input": "meeting.json", "speaker_at": "37.3=Sharad"}),
        Example("Several utterances at once, from the visual-ID review output",
                {"input": "meeting.json", "speaker_at": "1904.28=Carolyn,2269.98=Carolyn"}),
    ),
    "llm-correct": (
        Example("Claude fixes misheard names/jargon, using a glossary for context",
                {"input": "meeting.json", "glossary": "g.json"}),
        Example("Plain correction pass, no glossary",
                {"input": "meeting.json"}),
    ),
    "voiceprint-list": (
        Example("See who's enrolled and how many samples each has",
                {"store": "voiceprints.json"}),
    ),
    "voiceprint-enroll": (
        Example("Enroll everyone from an already-corrected transcript",
                {"transcript": "meeting.json", "media": "meeting.mp4",
                 "store": "voiceprints.json"}),
        Example("Enroll named people from the Desktop track of a muxed .mkv",
                {"transcript": "meeting.json", "media": "meeting.with-mic.mkv",
                 "store": "voiceprints.json", "names": "Ryan,Mar,Ness,John", "track": "1"}),
    ),
    "voiceprint-validate": (
        Example("Sanity-check the store against a confirmed transcript",
                {"transcript": "meeting.json", "media": "meeting.mp4",
                 "store": "voiceprints.json"}),
    ),
    "doctor": (
        Example("Run the full environment check (no arguments needed)",
                {}),
    ),
}

CATALOG: dict[str, Task] = {
    task.key: replace(task, examples=_EXAMPLES.get(task.key, ())) for task in _TASKS
}
