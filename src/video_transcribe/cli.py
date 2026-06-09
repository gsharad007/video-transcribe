"""Command-line interface: video/audio file -> clean, speaker-labeled transcript."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from video_transcribe import __version__, audio, diarize, formats, merge, punctuate
from video_transcribe.transcribe import Segment, load_model, transcribe

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


# faster-whisper shares the ~224-token prompt budget between hotwords AND the
# rolling previous-text context; too many hotwords leaves no room to decode
# ("maximum decoding length must be > 0"). Keep only the highest-value front
# slice -- the full list is better applied as post-hoc correction.
_MAX_HOTWORDS = 24


def _load_hotwords(args: argparse.Namespace) -> str | None:
    """Resolve --hotwords / --hotwords-file into a single comma-joined string."""
    terms: list[str] = []
    if args.hotwords_file:
        path = Path(args.hotwords_file)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(text)
            terms = list(data.get("hotwords", [])) if isinstance(data, dict) else list(data)
        else:
            terms = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if args.hotwords:
        terms = [t.strip() for t in args.hotwords.split(",") if t.strip()] + terms
    terms = terms[:_MAX_HOTWORDS]
    return ", ".join(terms) if terms else None


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
    p.add_argument("--hotwords", default=None, metavar="TERMS",
                   help="Comma-separated terms to bias recognition toward (names, jargon).")
    p.add_argument("--hotwords-file", default=None, metavar="PATH",
                   help="Hotwords from a file: a .json with a 'hotwords' array, or a text "
                        "file with one term per line.")
    p.add_argument("--speaker", default=None, metavar="NAME",
                   help="Label the whole transcript with one speaker name, for "
                        "single-presenter recordings (no diarization).")
    p.add_argument("--no-vad", action="store_true",
                   help="Disable voice-activity-detection filtering.")
    p.add_argument("--no-tidy", action="store_true",
                   help="Skip the light readability pass (capitalisation/spacing).")
    p.add_argument("--no-punctuate", action="store_true",
                   help="Skip ML punctuation/sentence restoration (needs 'readable' extra; "
                        "on by default when installed).")

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

    t = p.add_argument_group("multi-track input (e.g. ReLive 'Separate Microphone Track')")
    t.add_argument("--tracks", default=None, metavar="MAP",
                   help="Per-track speakers as IDX=NAME pairs, e.g. "
                        "--tracks \"0=Mar,1=Sharad\". Transcribes each audio track "
                        "separately and labels by track -- exact speakers, no diarization.")
    t.add_argument("--list-tracks", action="store_true",
                   help="List each input's audio tracks (index/codec/channels) and exit.")
    t.add_argument("--track-speakers", default=None, metavar="NAMES",
                   help="Treat the input FILES as parallel tracks of ONE recording, "
                        "labeled by these comma-separated names (e.g. \"Mar,Sharad\" for "
                        "video+separate-mic), merged by timestamp into one transcript.")
    t.add_argument("--mux", action="store_true",
                   help="With --track-speakers on a video+mic pair, also write one .mkv "
                        "with Mix (default) + Desktop + Mic audio tracks.")

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


def _transcribe_one(inp: Path, args: argparse.Namespace, fmts: list[str],
                    pipeline, punctuator) -> int:
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
            hotwords=args.hotwords_resolved,
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

    if punctuator is not None and not args.quiet:
        print("    restoring punctuation ...", file=sys.stderr)
    conv = merge.build_conversation(result.segments, turns,
                                    tidy=not args.no_tidy, punctuator=punctuator)
    if args.speaker and not turns:
        conv = replace(
            conv,
            speakers=[args.speaker],
            utterances=[replace(u, speaker=args.speaker) for u in conv.utterances],
            segments=[replace(s, speaker=args.speaker) for s in conv.segments],
        )
    meta = formats.Meta.from_result(inp.name, result,
                                    diarized=bool(turns) or bool(args.speaker))

    for fmt in fmts:
        out_path = out_dir / (inp.stem + EXT[fmt])
        out_path.write_text(formats.WRITERS[fmt](conv, meta), encoding="utf-8")
        _log(args.quiet, f"    wrote {out_path}")

    return 0


def _parse_tracks(spec: str) -> dict[int, str]:
    """Parse '0=Mar,1=Sharad' into {0: 'Mar', 1: 'Sharad'}."""
    mapping: dict[int, str] = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        idx, sep, name = part.partition("=")
        if not sep or not idx.strip().isdigit() or not name.strip():
            raise SystemExit(f"error: bad --tracks entry '{part}'. Use IDX=NAME, "
                             "e.g. --tracks \"0=Mar,1=Sharad\"")
        mapping[int(idx.strip())] = name.strip()
    return mapping


def _transcribe_tracks(inp: Path, args: argparse.Namespace, fmts: list[str],
                       model, punctuator, track_map: dict[int, str]) -> int:
    """Transcribe each audio track separately and label by track (no diarization)."""
    if not inp.exists():
        print(f"error: file not found: {inp}", file=sys.stderr)
        return 1
    out_dir = args.output_dir or inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    available = {s["a_index"] for s in audio.probe_streams(inp)}
    results: list[tuple[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="video-transcribe-") as tmp:
        for a_idx, name in sorted(track_map.items()):
            if a_idx not in available:
                print(f"warning: {inp.name} has no audio track {a_idx} "
                      f"(found {sorted(available) or 'none'}); skipping",
                      file=sys.stderr)
                continue
            wav = Path(tmp) / f"{inp.stem}.track{a_idx}.wav"
            _log(args.quiet, f"==> {inp.name}: track {a_idx} = {name}: extracting + transcribing ...")
            audio.extract_audio(inp, wav, stream_index=a_idx)
            result = transcribe(
                wav, model=model, model_size=args.model,
                language=args.language, vad_filter=not args.no_vad,
                hotwords=args.hotwords_resolved,
                progress=_make_progress(args.quiet, args.verbose),
            )
            if not args.quiet and not args.verbose:
                print("", file=sys.stderr)
            _log(args.quiet, f"    {name}: {len(result.segments)} segments over "
                             f"{result.duration:.0f}s")
            results.append((name, result))

    if not results:
        print(f"error: none of the requested tracks were found in {inp.name}", file=sys.stderr)
        return 1

    if punctuator is not None and not args.quiet:
        print("    restoring punctuation ...", file=sys.stderr)
    conv = merge.build_conversation_from_tracks(results, tidy=not args.no_tidy,
                                                punctuator=punctuator)
    meta = formats.Meta(
        title=inp.name,
        language=results[0][1].language,
        duration=max(r.duration for _, r in results),
        model=args.model,
        diarized=True,
    )
    for fmt in fmts:
        out_path = out_dir / (inp.stem + EXT[fmt])
        out_path.write_text(formats.WRITERS[fmt](conv, meta), encoding="utf-8")
        _log(args.quiet, f"    wrote {out_path}")
    return 0


def _transcribe_merged_files(inputs: list[Path], names: list[str],
                             args: argparse.Namespace, fmts: list[str],
                             model, punctuator) -> int:
    """Transcribe parallel track FILES of one recording, merge by timestamp.

    Used when the mic was captured to a separate file (not muxed). Both files
    share the recording's t=0 timeline, so segment timestamps line up directly.
    """
    if len(names) != len(inputs):
        print(f"error: --track-speakers has {len(names)} name(s) but {len(inputs)} "
              f"input file(s); they must match", file=sys.stderr)
        return 1
    for inp in inputs:
        if not inp.exists():
            print(f"error: file not found: {inp}", file=sys.stderr)
            return 1
    out_dir = args.output_dir or inputs[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="video-transcribe-") as tmp:
        for inp, name in zip(inputs, names):
            wav = Path(tmp) / (inp.stem + ".16k.wav")
            _log(args.quiet, f"==> {name}: {inp.name}: extracting + transcribing ...")
            audio.extract_audio(inp, wav)
            result = transcribe(
                wav, model=model, model_size=args.model,
                language=args.language, vad_filter=not args.no_vad,
                hotwords=args.hotwords_resolved,
                progress=_make_progress(args.quiet, args.verbose),
            )
            if not args.quiet and not args.verbose:
                print("", file=sys.stderr)
            _log(args.quiet, f"    {name}: {len(result.segments)} segments over "
                             f"{result.duration:.0f}s")
            results.append((name, result))

    if punctuator is not None and not args.quiet:
        print("    restoring punctuation ...", file=sys.stderr)
    conv = merge.build_conversation_from_tracks(results, tidy=not args.no_tidy,
                                                punctuator=punctuator)
    meta = formats.Meta(
        title=inputs[0].name,
        language=results[0][1].language,
        duration=max(r.duration for _, r in results),
        model=args.model,
        diarized=True,
    )
    for fmt in fmts:
        out_path = out_dir / (inputs[0].stem + EXT[fmt])
        out_path.write_text(formats.WRITERS[fmt](conv, meta), encoding="utf-8")
        _log(args.quiet, f"    wrote {out_path}")

    if args.mux:
        flags = [audio.has_video(i) for i in inputs]
        videos = [i for i, v in zip(inputs, flags) if v]
        mics = [i for i, v in zip(inputs, flags) if not v]
        if len(videos) == 1 and len(mics) == 1:
            mkv = out_dir / (videos[0].stem + ".with-mic.mkv")
            _log(args.quiet, f"    muxing video + mic -> {mkv.name} (copies the video) ...")
            audio.mux_tracks(videos[0], mics[0], mkv)
            _log(args.quiet, f"    wrote {mkv}")
        else:
            print(f"warning: --mux needs one video + one audio-only (mic) input "
                  f"(found {len(videos)} video / {len(mics)} audio); skipping mux",
                  file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    fmts = list(dict.fromkeys(args.formats or ["txt"]))
    track_map = _parse_tracks(args.tracks) if args.tracks else None
    track_speakers = ([n.strip() for n in args.track_speakers.split(",") if n.strip()]
                      if args.track_speakers else None)

    pipeline = None
    model = None
    try:
        if args.list_tracks:
            for inp in args.inputs:
                print(f"{inp.name}:")
                for s in audio.probe_streams(inp):
                    extra = f", title={s['title']}" if s.get("title") else ""
                    print(f"  track {s['a_index']}: {s['codec']}, {s['channels']}ch, "
                          f"{s['sample_rate']}Hz{extra}")
            return 0

        if track_map is not None or track_speakers is not None:
            if args.diarize:
                _log(args.quiet, "note: --diarize is ignored in track mode "
                                 "(speakers come from the tracks)")
            _log(args.quiet, f"loading model '{args.model}' "
                             f"({args.device}/{args.compute_type}) ...")
            model = load_model(args.model, args.device, args.compute_type)
        elif args.diarize:
            # Load + authenticate up front so a bad token fails before any slow ASR.
            token = args.hf_token or os.environ.get("HF_TOKEN")
            _log(args.quiet, f"loading diarization model '{args.diarize_model}' ...")
            pipeline = diarize.load_pipeline(args.diarize_model, hf_token=token, num_threads=None)

        punctuator = None
        if not args.no_punctuate and punctuate.available():
            _log(args.quiet, "loading punctuation model (pcs_en) ...")
            pmodel = punctuate.load()
            punctuator = lambda texts: punctuate.restore(pmodel, texts)

        args.hotwords_resolved = _load_hotwords(args)
        if args.hotwords_resolved:
            n = args.hotwords_resolved.count(",") + 1
            _log(args.quiet, f"biasing recognition with {n} hotwords (front-loaded)")

        if track_speakers is not None:
            return _transcribe_merged_files(args.inputs, track_speakers, args, fmts,
                                            model, punctuator)

        rc = 0
        for inp in args.inputs:
            if track_map is not None:
                rc |= _transcribe_tracks(inp, args, fmts, model, punctuator, track_map)
            else:
                rc |= _transcribe_one(inp, args, fmts, pipeline, punctuator)
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
