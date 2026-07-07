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
from dataclasses import dataclass, field
from typing import Literal

from video_transcribe.diarize import DEFAULT_DIARIZE_MODEL
from video_transcribe.voiceprint import DEFAULT_MATCH_THRESHOLD

__all__ = (
    "Arg",
    "Task",
    "CATALOG",
    "CATEGORY_ORDER",
    "CATEGORY_LABELS",
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

    @property
    def positional(self) -> bool:
        return self.flag is None


@dataclass(frozen=True, slots=True)
class Task:
    key: str
    label: str
    category: str
    summary: str
    argv_prefix: tuple[str, ...]
    args: tuple[Arg, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)


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
        return [arg.flag] if value else []
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
    flags: list[str] = []
    positionals: list[str] = []
    for arg in task.args:
        raw = values.get(arg.name, arg.default)
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
MODEL = Arg("model", "str", "Whisper model.", flag="--model", default="large-v3",
            placeholder="large-v3 | large-v3-turbo | base")
LANGUAGE = Arg("language", "str", "Language code (blank = autodetect).", flag="--language",
               placeholder="en, ko, ...")
DEVICE = Arg("device", "choice", "Inference device (cuda = NVIDIA only).", flag="--device",
             default="cpu", choices=("cpu", "cuda", "auto"))
COMPUTE = Arg("compute_type", "str", "CTranslate2 compute type.", flag="--compute-type",
              default="int8", placeholder="int8 | float16")
HOTWORDS = Arg("hotwords", "str", "Bias terms (names/jargon), comma-separated.", flag="--hotwords",
               placeholder="GrimeReaper, pyannote")
HOTWORDS_FILE = Arg("hotwords_file", "path", "Hotwords file (.json or one term per line).",
                    flag="--hotwords-file")
OUTPUT_DIR = Arg("output_dir", "path", "Output directory (blank = beside input).", flag="--output-dir")
SPEAKER = Arg("speaker", "str", "Label the whole transcript with one speaker name.", flag="--speaker")
NO_VAD = Arg("no_vad", "bool", "Disable voice-activity-detection filtering.", flag="--no-vad")
NO_TIDY = Arg("no_tidy", "bool", "Skip the light readability pass.", flag="--no-tidy")
NO_PUNCT = Arg("no_punctuate", "bool", "Skip ML punctuation/sentence restoration.", flag="--no-punctuate")
KEEP_AUDIO = Arg("keep_audio", "bool", "Keep the intermediate 16 kHz WAV.", flag="--keep-audio")
VERBOSE = Arg("verbose", "bool", "Stream each segment as it is decoded.", flag="--verbose")
QUIET = Arg("quiet", "bool", "Suppress progress output.", flag="--quiet")

HF_TOKEN = Arg("hf_token", "str", "Hugging Face token (else uses $HF_TOKEN).", flag="--hf-token")
SPEAKERS = Arg("speakers", "int", "Exact number of speakers, if known.", flag="--speakers")
MIN_SPK = Arg("min_speakers", "int", "Lower bound on speakers.", flag="--min-speakers")
MAX_SPK = Arg("max_speakers", "int", "Upper bound on speakers.", flag="--max-speakers")
DIARIZE_MODEL = Arg("diarize_model", "str", "pyannote pipeline.", flag="--diarize-model",
                    default=DEFAULT_DIARIZE_MODEL)
VOICEPRINTS = Arg("voiceprints", "path", "Voiceprint store JSON (auto-name speakers by voice).",
                  flag="--voiceprints")
VOICE_THRESHOLD = Arg("voice_threshold", "float", "Cosine-similarity threshold for a voice match.",
                      flag="--voice-threshold", default=str(DEFAULT_MATCH_THRESHOLD))

# The shared trailing block reused by every transcription mode, in a sensible
# tab order (output shape first, then quality knobs, then flags).
_COMMON_TAIL = (FORMAT, MODEL, LANGUAGE, DEVICE, COMPUTE, HOTWORDS, HOTWORDS_FILE,
                OUTPUT_DIR, NO_VAD, NO_TIDY, NO_PUNCT, KEEP_AUDIO, VERBOSE, QUIET)

# --------------------------------------------------------------------------- #
# the catalog
# --------------------------------------------------------------------------- #

CATEGORY_ORDER = ("transcribe", "correct", "speakers", "media", "setup")
CATEGORY_LABELS = {
    "transcribe": "Transcribe",
    "correct": "Correct & clean",
    "speakers": "Voiceprints",
    "media": "Media",
    "setup": "Setup & checks",
}

_TASKS: tuple[Task, ...] = (
    Task(
        key="transcribe",
        label="Transcribe (basic)",
        category="transcribe",
        summary="The everyday job: transcribe one or more files to a readable transcript, "
                "no speaker labels. Output lands beside each input (or in --output-dir). "
                "Add --speaker to tag a single-presenter recording with one name.",
        argv_prefix=("-m", "video_transcribe"),
        args=(INPUTS, SPEAKER, *_COMMON_TAIL),
        tags=("transcribe", "whisper", "basic"),
    ),
    Task(
        key="diarize",
        label="Transcribe + diarize speakers",
        category="transcribe",
        summary="Transcribe and label who-said-what with pyannote. Needs a Hugging Face token "
                "(--hf-token or $HF_TOKEN) and the 'diarize' extra. Pass --speakers if you know "
                "the count; point --voiceprints at a store to auto-name known voices.",
        argv_prefix=("-m", "video_transcribe", "--diarize"),
        args=(INPUTS, SPEAKERS, MIN_SPK, MAX_SPK, HF_TOKEN, DIARIZE_MODEL,
              VOICEPRINTS, VOICE_THRESHOLD, *_COMMON_TAIL),
        tags=("transcribe", "diarize", "speakers", "pyannote"),
    ),
    Task(
        key="list-tracks",
        label="List audio tracks",
        category="transcribe",
        summary="Print each input's audio tracks (index / codec / channels) and exit. Run this "
                "first to find the track indices for the by-track modes below.",
        argv_prefix=("-m", "video_transcribe", "--list-tracks"),
        args=(INPUTS,),
        tags=("inspect", "tracks"),
    ),
    Task(
        key="tracks-in-file",
        label="Transcribe by track (one file)",
        category="transcribe",
        summary="Multi-track file (e.g. ReLive mic muxed into the video): transcribe each track "
                "separately and label by track -- exact speakers, no diarization. Map with "
                "--tracks like \"0=Mar,1=Sharad\" (see List audio tracks for indices).",
        argv_prefix=("-m", "video_transcribe"),
        args=(
            INPUTS,
            Arg("tracks", "str", "Per-track speakers as IDX=NAME pairs.", flag="--tracks",
                required=True, placeholder="0=Mar,1=Sharad"),
            FORMAT, MODEL, LANGUAGE, HOTWORDS, OUTPUT_DIR, NO_TIDY, NO_PUNCT, VERBOSE, QUIET,
        ),
        tags=("transcribe", "tracks", "speakers"),
    ),
    Task(
        key="tracks-files",
        label="Transcribe parallel track files",
        category="transcribe",
        summary="Separate files for one recording (e.g. video + separate mic .m4a): transcribe "
                "each and merge by timestamp, labeled by --track-speakers like \"Mar,Sharad\" "
                "(one name per file, in order). --mux also writes a combined .mkv.",
        argv_prefix=("-m", "video_transcribe"),
        args=(
            INPUTS,
            Arg("track_speakers", "str", "Comma-separated speaker names, one per input file.",
                flag="--track-speakers", required=True, placeholder="Mar,Sharad"),
            Arg("mux", "bool", "Also write a combined .mkv (Mix + Desktop + Mic).", flag="--mux"),
            FORMAT, MODEL, LANGUAGE, HOTWORDS, OUTPUT_DIR, NO_TIDY, NO_PUNCT, VERBOSE, QUIET,
        ),
        tags=("transcribe", "tracks", "mux", "speakers"),
    ),
    Task(
        key="hybrid",
        label="Diarize one file + fixed tracks",
        category="transcribe",
        summary="Group call + your own mic: acoustically diarize one input (--diarize-track IDX) "
                "while the other file(s) are fixed single-speaker tracks named by --track-speakers. "
                "Diarized speakers come out generic (Speaker 1...); rename later with Correct.",
        argv_prefix=("-m", "video_transcribe"),
        args=(
            INPUTS,
            Arg("diarize_track", "int", "0-based index of the input file to diarize.",
                flag="--diarize-track", required=True, placeholder="0"),
            Arg("track_speakers", "str", "Names for the OTHER (non-diarized) files, in order.",
                flag="--track-speakers", required=True, placeholder="Sharad"),
            SPEAKERS, MIN_SPK, MAX_SPK, HF_TOKEN, VOICEPRINTS, VOICE_THRESHOLD,
            Arg("mux", "bool", "Also write a combined .mkv (Mix + Desktop + Mic).", flag="--mux"),
            FORMAT, MODEL, LANGUAGE, HOTWORDS, OUTPUT_DIR, NO_TIDY, NO_PUNCT, VERBOSE, QUIET,
        ),
        tags=("transcribe", "diarize", "tracks", "hybrid", "speakers"),
    ),
    Task(
        key="correct",
        label="Glossary correction (local)",
        category="correct",
        summary="Apply a glossary (term fixes + speaker names) to a finished transcript .json -- "
                "deterministic, fully local, no re-transcription. Re-emits txt/srt/vtt/json beside "
                "the input. Use --speakers to override the glossary's speaker map for one run.",
        argv_prefix=("-m", "video_transcribe.correct"),
        args=(
            Arg("input", "path", "Transcript .json produced by video-transcribe.", required=True),
            Arg("glossary", "path", "Glossary JSON (speaker_map + corrections).", flag="--glossary",
                required=True),
            Arg("speakers", "str", "Override speaker map, e.g. 'Speaker 1=JV,Speaker 2=Sharad'.",
                flag="--speakers"),
            OUTPUT_DIR,
            Arg("format", "str", "Output formats (default: txt,srt,json).", flag="--format",
                choices=_FMT_CHOICES, repeatable=True, placeholder="txt,srt,json"),
        ),
        tags=("correct", "glossary", "local"),
    ),
    Task(
        key="llm-correct",
        label="LLM correction (Claude API)",
        category="correct",
        summary="Correction-only pass over a transcript .json via the Claude API: fixes misheard "
                "names/jargon and obvious ASR slips, 1:1 with the input, never rewrites. Writes "
                "separate .llm.* files + a .llm-changes.txt diff. Sends text off-machine; needs the "
                "'llm' extra + ANTHROPIC_API_KEY.",
        argv_prefix=("-m", "video_transcribe.llm_correct"),
        args=(
            Arg("input", "path", "Transcript .json produced by video-transcribe.", required=True),
            Arg("glossary", "path", "Optional glossary JSON for context.", flag="--glossary"),
            Arg("model", "str", "Claude model.", flag="--model", default="claude-opus-4-8"),
            OUTPUT_DIR,
            Arg("format", "str", "Output formats (default: txt,json).", flag="--format",
                choices=("txt", "json"), repeatable=True, placeholder="txt,json"),
        ),
        tags=("correct", "llm", "claude", "api"),
    ),
    Task(
        key="voiceprint-list",
        label="Voiceprints: list",
        category="speakers",
        summary="List enrolled people in a voiceprint store and how many samples each has.",
        argv_prefix=("-m", "video_transcribe.voiceprint", "list"),
        args=(Arg("store", "path", "Voiceprint store JSON.", flag="--store", required=True),),
        tags=("voiceprint", "list"),
    ),
    Task(
        key="voiceprint-enroll",
        label="Voiceprints: enroll",
        category="speakers",
        summary="Grow a voiceprint store from an already-corrected transcript .json + its source "
                "media, so future diarized recordings auto-name these voices. Use --names to enroll "
                "only some speakers, --track for one track of a multi-track file.",
        argv_prefix=("-m", "video_transcribe.voiceprint", "enroll"),
        args=(
            Arg("transcript", "path", "Corrected transcript .json (real speaker names).", required=True),
            Arg("media", "path", "Audio/video file that speech came from.", required=True),
            Arg("store", "path", "Voiceprint store JSON (created/updated).", flag="--store", required=True),
            Arg("names", "str", "Only enroll these speakers (comma-separated; default: all).", flag="--names"),
            Arg("track", "int", "0-based audio track index for a multi-track file.", flag="--track"),
            HF_TOKEN,
        ),
        tags=("voiceprint", "enroll"),
    ),
    Task(
        key="voiceprint-validate",
        label="Voiceprints: validate",
        category="speakers",
        summary="Check a store's matches against a transcript whose speaker names you've already "
                "confirmed, without changing the store -- a sanity check before trusting it.",
        argv_prefix=("-m", "video_transcribe.voiceprint", "validate"),
        args=(
            Arg("transcript", "path", "Corrected transcript .json (real speaker names).", required=True),
            Arg("media", "path", "Audio/video file that speech came from.", required=True),
            Arg("store", "path", "Voiceprint store JSON.", flag="--store", required=True),
            Arg("names", "str", "Only validate these speakers (comma-separated).", flag="--names"),
            Arg("track", "int", "0-based audio track index for a multi-track file.", flag="--track"),
            Arg("threshold", "float", "Match threshold.", flag="--threshold",
                default=str(DEFAULT_MATCH_THRESHOLD)),
            HF_TOKEN,
        ),
        tags=("voiceprint", "validate"),
    ),
    Task(
        key="mux",
        label="Mux video + mic -> MKV",
        category="media",
        summary="Combine a video (with its desktop audio) + a separate mic file into one .mkv: a "
                "default Mix track plus isolated Desktop and Mic tracks. Video is stream-copied "
                "(no re-encode). No transcript is produced.",
        argv_prefix=("-m", "video_transcribe.mux"),
        args=(
            Arg("video", "path", "Video file (with desktop/system audio).", required=True),
            Arg("mic", "path", "Separate microphone audio file.", required=True),
            Arg("output", "path", "Output .mkv (blank = <video>.with-mic.mkv).", flag="--output"),
        ),
        tags=("media", "mux", "ffmpeg"),
    ),
    Task(
        key="doctor",
        label="Environment doctor",
        category="setup",
        summary="Check the local setup: ffmpeg/ffprobe on PATH, the optional extras "
                "(diarize / readable / llm), and the tokens diarization and LLM correction need. "
                "Run this first if a job fails to start.",
        argv_prefix=("-m", "video_transcribe.tui_doctor"),
        args=(),
        tags=("setup", "doctor", "check"),
    ),
)

CATALOG: dict[str, Task] = {task.key: task for task in _TASKS}
