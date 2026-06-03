"""Command-line interface: video/audio file -> transcript files."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from video_transcribe import __version__, audio, formats
from video_transcribe.transcribe import Segment, transcribe

EXT = {"txt": ".txt", "srt": ".srt", "vtt": ".vtt", "json": ".json"}


def _fmt_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _make_progress(quiet: bool, verbose: bool):
    if quiet:
        return None

    width = 30

    def cb(seg: Segment, total: float) -> None:
        if verbose:
            print(f"  [{_fmt_ts(seg.start)} -> {_fmt_ts(seg.end)}] {seg.text.strip()}",
                  file=sys.stderr)
            return
        frac = min(seg.end / total, 1.0) if total else 0.0
        filled = int(frac * width)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r  transcribing [{bar}] {frac * 100:5.1f}%  "
              f"{_fmt_ts(seg.end)}/{_fmt_ts(total)} ",
              end="", file=sys.stderr, flush=True)

    return cb


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video-transcribe",
        description="Transcribe speech from video or audio files locally using "
                    "faster-whisper. Audio is extracted with ffmpeg, so anything "
                    "ffmpeg can read works.",
    )
    p.add_argument("inputs", nargs="+", type=Path, metavar="FILE",
                   help="One or more video/audio files to transcribe.")
    p.add_argument("-f", "--format", dest="formats", action="append",
                   choices=sorted(EXT), metavar="FMT",
                   help="Output format(s): txt, srt, vtt, json. Repeatable "
                        "(default: txt).")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="Directory for output files (default: next to each input).")
    p.add_argument("--model", default="large-v3-turbo",
                   help="Whisper model name/size (default: large-v3-turbo). "
                        "Use 'base' or 'small' for quick tests.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                   help="Inference device (default: cpu). faster-whisper has no "
                        "AMD/ROCm backend, so 'cuda' is NVIDIA-only.")
    p.add_argument("--compute-type", default="int8",
                   help="CTranslate2 compute type (default: int8, best on CPU; "
                        "try float16 on CUDA).")
    p.add_argument("--language", default=None,
                   help="Spoken-language code, e.g. en, fr (default: autodetect).")
    p.add_argument("--no-vad", action="store_true",
                   help="Disable voice-activity-detection filtering.")
    p.add_argument("--keep-audio", action="store_true",
                   help="Keep the intermediate 16 kHz WAV alongside the output.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Stream each segment as it is decoded.")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Suppress progress output.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg, file=sys.stderr)


def _transcribe_one(inp: Path, args: argparse.Namespace, fmts: list[str]) -> int:
    if not inp.exists():
        print(f"error: file not found: {inp}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="video-transcribe-") as tmp:
        wav = Path(tmp) / (inp.stem + ".16k.wav")
        _log(args.quiet, f"==> {inp.name}: extracting audio with ffmpeg ...")
        audio.extract_audio(inp, wav)

        if args.keep_audio:
            kept = out_dir / wav.name
            kept.write_bytes(wav.read_bytes())
            _log(args.quiet, f"    kept audio: {kept}")

        _log(args.quiet,
             f"    loading model '{args.model}' "
             f"({args.device}/{args.compute_type}) - first run downloads it ...")

        result = transcribe(
            wav,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            vad_filter=not args.no_vad,
            progress=_make_progress(args.quiet, args.verbose),
        )

    if not args.quiet and not args.verbose:
        print("", file=sys.stderr)  # terminate the progress line
    _log(args.quiet,
         f"    detected {result.language} "
         f"({result.language_probability * 100:.0f}% conf), "
         f"{len(result.segments)} segments over {result.duration:.0f}s")

    for fmt in fmts:
        out_path = out_dir / (inp.stem + EXT[fmt])
        out_path.write_text(formats.WRITERS[fmt](result), encoding="utf-8")
        _log(args.quiet, f"    wrote {out_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # de-dupe formats while preserving the order they were given in
    fmts = list(dict.fromkeys(args.formats or ["txt"]))

    try:
        rc = 0
        for inp in args.inputs:
            rc |= _transcribe_one(inp, args, fmts)
        return rc
    except audio.FFmpegNotFound as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
