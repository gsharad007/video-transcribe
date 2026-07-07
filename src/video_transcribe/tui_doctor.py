"""Environment doctor: is the machine set up to run each part of the pipeline?

Reports what's required (ffmpeg + faster-whisper -- the base transcription path)
versus what's optional (the diarize / readable / llm extras and their tokens),
so a user can tell at a glance why a job won't start and exactly which
``uv sync --extra ...`` or token they're missing. No Textual/torch import, so it
runs standalone (``python -m video_transcribe.tui_doctor``) and is importable in
tests.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass

__all__ = ("Check", "check_environment", "main")


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    required: bool
    detail: str
    hint: str = ""


def _module_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # A broken/partial install can raise rather than return None.
        return False


def check_environment() -> list[Check]:
    checks: list[Check] = []

    py_ok = sys.version_info >= (3, 10)
    checks.append(Check(
        "python", py_ok, True,
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "" if py_ok else "video-transcribe needs Python >= 3.10.",
    ))

    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        checks.append(Check(
            tool, path is not None, True,
            path or "not found on PATH",
            "" if path else "Install ffmpeg and put it on PATH: https://ffmpeg.org/download.html",
        ))

    checks.append(Check(
        "faster-whisper", _module_present("faster_whisper"), True,
        "installed" if _module_present("faster_whisper") else "missing",
        "" if _module_present("faster_whisper") else "Run: uv sync",
    ))

    diarize_ok = _module_present("torch") and _module_present("pyannote.audio")
    checks.append(Check(
        "diarize extra", diarize_ok, False,
        "installed (torch + pyannote)" if diarize_ok else "missing",
        "" if diarize_ok else "For --diarize / voiceprints: uv sync --extra diarize",
    ))

    readable_ok = _module_present("punctuators")
    checks.append(Check(
        "readable extra", readable_ok, False,
        "installed (punctuators)" if readable_ok else "missing",
        "" if readable_ok else "For ML punctuation: uv sync --extra readable",
    ))

    llm_ok = _module_present("anthropic")
    checks.append(Check(
        "llm extra", llm_ok, False,
        "installed (anthropic)" if llm_ok else "missing",
        "" if llm_ok else "For LLM correction: uv sync --extra llm",
    ))

    tui_ok = _module_present("textual") and _module_present("psutil")
    checks.append(Check(
        "tui extra", tui_ok, False,
        "installed (textual + psutil)" if tui_ok else "missing",
        "" if tui_ok else "For this TUI: uv sync --extra tui",
    ))

    hf = bool(os.environ.get("HF_TOKEN"))
    checks.append(Check(
        "HF_TOKEN", hf, False,
        "set" if hf else "not set",
        "" if hf else "Diarization needs a Hugging Face token (env HF_TOKEN or `hf auth login`).",
    ))

    anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    checks.append(Check(
        "ANTHROPIC_API_KEY", anthropic_key, False,
        "set" if anthropic_key else "not set",
        "" if anthropic_key else "LLM correction needs ANTHROPIC_API_KEY (or `ant auth login`).",
    ))

    return checks


def main(argv: list[str] | None = None) -> int:
    checks = check_environment()
    width = max(len(c.name) for c in checks)
    print("video-transcribe environment\n" + "=" * 40)
    for c in checks:
        if c.ok:
            mark = "ok  "
        elif c.required:
            mark = "FAIL"
        else:
            mark = "--  "
        tag = "" if c.required else "  (optional)"
        print(f"  [{mark}] {c.name:<{width}}  {c.detail}{tag}")
        if not c.ok and c.hint:
            print(f"         -> {c.hint}")
    missing_required = [c for c in checks if c.required and not c.ok]
    print("-" * 40)
    if missing_required:
        names = ", ".join(c.name for c in missing_required)
        print(f"missing required: {names}")
        return 1
    print("base transcription path is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
