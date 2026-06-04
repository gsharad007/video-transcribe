"""Command-line interface: video/audio file -> clean, speaker-labeled transcript."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from video_transcribe import __version__, audio, diarize, formats, merge
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
        description="Transcribe speech from video/audio locally with faster-whisper, "
                    "with optional speaker diarization. Audio is extracted via ffmpeg.",
    )
    p.add_argument("inputs", nargs="+", type=Path, metavar="FILE",
                   help="One or more video/audio files to transcribe.")
    p.add_argument("-f", "--format", dest="formats", action="append",
                   choices=sorted(EXT), metavar="FMT",
                   help="Output format(s): txt, srt, vtt, json. Repeatable "
                        "(default: txt).")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="Directory for output files (default: next to each input).")
    p.add_argument("--model", default="large-v3",
                   help="Whisper model (default: large-v3 = highest quality). "
                        "Use 'large-v3-turbo' for ~speed, 'base' for quick tests.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                   help="Inference device (default: cpu). 'cuda' is NVIDIA-only.")
    p.add_argument("--compute-type", default="int8",
                   help="CTranslate2 compute type (default: int8; try float16 on CUDA).")
    p.add_argument("--language", default=None,
                   help="Spoken-language code, e.g. en, ko (default: autodetect).")
    p.add_argument("--no-vad", action="store_true",
                   help="Disable voice-activity-detection filtering.")
    p.add_argument("--no-tidy", action="store_true",
                   help="Skip the light readability pass (capitalisation/spacing).")

    g = p.add_argument_group("speaker diarization (who-said-what)")
    g.add_argument("--diarize", action="store_true",
                   help="Label speakers using pyannote (needs --hf-token / HF_TOKEN).")
    g.add_argument("--hf-token", default=None,
                   help="Hugging Face token (else read from HF_TOKEN env var).")
    g.add_argument("--diarize-model", default=diarize.DEFAULT_DIARIZE_MODEL,
                   help=f"pyannote pipeline (default: {diarize.DEFAULT_DIARIZE_MODEL}).")
    g.add_argument("--speakers", type=int, default=None,
                   help="Exact number of speakers, if known.")
    g.add_argument("--min-speakers", type=int, default=None, help="Lower bound on speakers.")
    g.add_argument("--max-speakers", type=int, default=None, help="Upper bound on speakers.")

    p.add_argument("--keep-audio", action="store_true",
                   help="Keep the intermediate 16 kHz WAV alongside the output.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Stream each segment as it is decoded.")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg, file=sys.stderr)


def _transcribe_one(inp: Path, args: argparse.Namespace, fmts: list[str], pipeline) -> int:
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
            (out_dir / wav.name).write_bytes(wav.read_bytes())

        _log(args.quiet,
             f"    transcribing with '{args.model}' ({args.device}/{args.compute_type})"
             " - first run downloads the model ...")
        result = transcribe(
            wav,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            vad_filter=not args.no_vad,
            word_timestamps=args.diarize,
            progress=_make_progress(args.quiet, args.verbose),
        )
        if not args.quiet and not args.verbose:
            print("", file=sys.stderr)  # terminate the progress line
        _log(args.quiet,
             f"    detected {result.language} "
             f"({result.language_probability * 100:.0f}% conf), "
             f"{len(result.segments)} segments over {result.duration:.0f}s")

        turns = []
        if pipeline is not None:
            _log(args.quiet, "    diarizing speakers (slow on CPU) ...")
            turns = diarize.run_pipeline(
                pipeline, wav,
                num_speakers=args.speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
            n = len({t.speaker for t in turns})
            _log(args.quiet, f"    found {n} speaker(s) across {len(turns)} turns")

    conv = merge.build_conversation(result.segments, turns, tidy=not args.no_tidy)
    meta = formats.Meta.from_result(inp.name, result, diarized=bool(turns))

    for fmt in fmts:
        out_path = out_dir / (inp.stem + EXT[fmt])
        out_path.write_text(formats.WRITERS[fmt](conv, meta), encoding="utf-8")
        _log(args.quiet, f"    wrote {out_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    fmts = list(dict.fromkeys(args.formats or ["txt"]))

    pipeline = None
    try:
        if args.diarize:
            # Load + authenticate up front so a bad token fails before any slow ASR.
            token = args.hf_token or os.environ.get("HF_TOKEN")
            _log(args.quiet, f"loading diarization model '{args.diarize_model}' ...")
            pipeline = diarize.load_pipeline(
                args.diarize_model, hf_token=token,
                num_threads=None,
            )

        rc = 0
        for inp in args.inputs:
            rc |= _transcribe_one(inp, args, fmts, pipeline)
        return rc
    except audio.FFmpegNotFound as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except diarize.DiarizationError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
