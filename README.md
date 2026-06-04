# video-transcribe

Transcribe speech from video/audio files **locally** — no cloud, no API keys for
the transcription itself. Audio is extracted with `ffmpeg`, transcribed with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), and optionally
**speaker-labeled** with [pyannote.audio](https://github.com/pyannote/pyannote-audio).
Output is clean, timestamped, speaker-grouped text (plus SRT/VTT/JSON).

## Requirements

- Python ≥ 3.10
- [`ffmpeg`](https://ffmpeg.org/download.html) on your `PATH` (`ffmpeg` + `ffprobe`)

## Install

```pwsh
uv sync                    # transcription only
uv sync --extra diarize    # + speaker diarization (pulls in torch + pyannote)
```

## Usage

```pwsh
# Highest quality (large-v3), readable transcript next to the input
uv run video-transcribe talk.mp4

# With speaker labels (needs a Hugging Face token — see below)
uv run video-transcribe meeting.mp4 --diarize -f txt -f srt

# Tell pyannote how many speakers if you know (more accurate)
uv run video-transcribe interview.mp4 --diarize --speakers 2

# Faster, lower quality; force language; quick test
uv run video-transcribe clip.mp4 --model large-v3-turbo --language en
```

Output `.txt` looks like:

```
meeting.mp4
Language: en  |  Duration: 31:54  |  Model: large-v3  |  Speakers: 3
====================================================================

[00:00] Speaker 1: Sorry I'm one minute late, I needed 30 seconds between calls.

[00:05] Speaker 2: I totally understand — you probably have a jam-packed day.
```

Run `uv run video-transcribe --help` for all options.

| Option           | Default          | Notes                                              |
|------------------|------------------|----------------------------------------------------|
| `--model`        | `large-v3`       | highest quality; `large-v3-turbo` ~speed; `base` test |
| `--diarize`      | off              | speaker labels via pyannote (needs HF token)       |
| `--hf-token`     | `$env:HF_TOKEN`  | Hugging Face token for the gated diarization model |
| `--speakers`     | auto             | exact count; or `--min/--max-speakers`             |
| `--device`       | `cpu`            | `cuda` is **NVIDIA-only**                          |
| `--language`     | autodetect       | force a code like `en`/`ko` to skip detection      |
| `-f/--format`    | `txt`            | repeatable: `txt`, `srt`, `vtt`, `json`            |
| `--no-tidy`      | off              | keep raw casing/spacing (skip readability pass)    |

First run downloads the model (large-v3 ≈ 3 GB) to `~/.cache/huggingface`.

## Speaker diarization setup (one time)

The diarization model is **gated** on Hugging Face, so you need a free token and
must accept its terms once:

1. Create a token (read scope): <https://huggingface.co/settings/tokens>
2. Accept the terms (click "Agree and access") on the default model:
   - <https://huggingface.co/pyannote/speaker-diarization-community-1>

   (pyannote.audio 4.x uses `community-1`; older `speaker-diarization-3.1` pulls
   from it too. If a load error names another gated repo, accept that one as well.)
3. Authenticate, then run:
   ```pwsh
   uv run hf auth login                # caches the token locally (recommended)
   uv run video-transcribe meeting.mp4 --diarize
   # ...or, without logging in:
   uv run video-transcribe meeting.mp4 --diarize --hf-token hf_xxx
   ```

Diarization runs on CPU here and is the slow part — budget several minutes for a
long meeting. Speakers are auto-named `Speaker 1`, `Speaker 2`, … in order of
first appearance; rename them in the output as you like.

## GPU note (AMD / this machine)

Transcription runs on **CPU** — on a Ryzen 9950X3D that's plenty fast.
CTranslate2 (faster-whisper) and PyTorch here are **CPU/NVIDIA-only**, so the
Radeon R9700 isn't used. The GPU path is whisper.cpp built with **Vulkan**
(cross-vendor, Windows-friendly); the backend boundary lives in `transcribe.py`
so it can slot in without touching the CLI, merge, or formatters.

## Layout

```
src/video_transcribe/
  audio.py        # ffmpeg: probe duration, extract 16 kHz mono WAV
  transcribe.py   # faster-whisper wrapper -> Segment / Word / TranscriptionResult
  diarize.py      # pyannote wrapper -> SpeakerTurn (in-memory WAV, no torchcodec)
  merge.py        # assign speakers, clean hallucinations, regroup -> Conversation
  formats.py      # readable txt + srt / vtt / json writers
  cli.py          # argument parsing + orchestration
tests/
  smoke.py        # model-free merge/format checks
  real_check.py   # faster-whisper word-timestamps -> merge (no HF token)
```
