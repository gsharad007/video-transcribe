# video-transcribe

Transcribe speech from video/audio files **locally** — no cloud, no API keys.
Audio is extracted with `ffmpeg`, then transcribed with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2).

## Requirements

- Python ≥ 3.10
- [`ffmpeg`](https://ffmpeg.org/download.html) on your `PATH` (`ffmpeg` + `ffprobe`)

## Install

```pwsh
uv sync          # creates .venv and installs everything
```

(or `pip install -e .` into a venv of your choice)

## Usage

```pwsh
# Simplest: write talk.txt next to the input
uv run video-transcribe talk.mp4

# SRT + VTT subtitles, into a folder
uv run video-transcribe lecture.mkv -f srt -f vtt -o out/

# Fast smoke test with a tiny model, English forced, stream segments live
uv run video-transcribe clip.mp4 --model base --language en -v

# Many files, JSON with word timings + plain text
uv run video-transcribe *.mp4 -f json -f txt
```

Run `uv run video-transcribe --help` for all options.

| Option           | Default          | Notes                                            |
|------------------|------------------|--------------------------------------------------|
| `--model`        | `large-v3-turbo` | `tiny`/`base`/`small`/`medium`/`large-v3`/`turbo`|
| `--device`       | `cpu`            | `cuda` is **NVIDIA-only** (see GPU note below)   |
| `--compute-type` | `int8`           | `int8` on CPU; `float16` on CUDA                 |
| `--language`     | autodetect       | force with a code like `en` to skip detection    |
| `-f/--format`    | `txt`            | repeatable: `txt`, `srt`, `vtt`, `json`          |

The first run downloads the model from Hugging Face (large-v3-turbo ≈ 1.5 GB),
cached under `~/.cache/huggingface`. Later runs are offline.

## GPU note (AMD / this machine)

`faster-whisper` runs on **CPU here** — and on a Ryzen 9950X3D that's plenty
fast. CTranslate2 has **no AMD/ROCm backend**, so `--device cuda` only helps on
NVIDIA cards.

To actually use the **Radeon R9700 (RDNA4)** for inference, the path is
**whisper.cpp built with Vulkan** (cross-vendor, works on Windows) and a binding
such as `pywhispercpp`/`whisper-cpp-python`, or the Rust crate `whisper-rs` with
its `vulkan` feature. That's a heavier native build (needs the Vulkan SDK); the
backend boundary lives in `transcribe.py` so it can be added without touching
the CLI or output formatters.

## Layout

```
src/video_transcribe/
  audio.py        # ffmpeg: probe duration, extract 16 kHz mono WAV
  transcribe.py   # faster-whisper wrapper -> Segment / TranscriptionResult
  formats.py      # txt / srt / vtt / json writers
  cli.py          # argument parsing + orchestration
```
