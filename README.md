# video-transcribe

Transcribe speech from video/audio files **locally** — no cloud, no API keys for
the transcription itself. Audio is extracted with `ffmpeg`, transcribed with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), optionally
**speaker-labeled** with [pyannote.audio](https://github.com/pyannote/pyannote-audio),
and **re-punctuated** (sentences + capitalization) with
[punctuators](https://github.com/1-800-BAD-CODE/punctuators) so Whisper's run-on
output reads cleanly. Result: timestamped, speaker-grouped paragraphs (plus
SRT/VTT/JSON).

## Requirements

- Python ≥ 3.10
- [`ffmpeg`](https://ffmpeg.org/download.html) on your `PATH` (`ffmpeg` + `ffprobe`)

## Install

```pwsh
uv sync                                   # transcription only
uv sync --extra diarize --extra readable  # + speaker labels + punctuation (full)
```

- `diarize` pulls in torch + pyannote (~2–3 GB); needed for `--diarize`.
- `readable` pulls in `punctuators` (tiny — reuses the onnxruntime faster-whisper
  already installs); enables sentence/punctuation restoration. On by default when
  installed; disable per-run with `--no-punctuate`.

## Usage

```pwsh
# Highest quality (large-v3), readable transcript next to the input
uv run video-transcribe talk.mp4

# With speaker labels (needs a Hugging Face token — see below)
uv run video-transcribe meeting.mp4 --diarize -f txt -f srt

# Tell pyannote how many speakers if you know (more accurate)
uv run video-transcribe interview.mp4 --diarize --speakers 2

# Two-track recording (e.g. ReLive separate mic): exact speakers, no diarization
uv run video-transcribe meeting.mp4 --list-tracks            # see track indices
uv run video-transcribe meeting.mp4 --tracks "0=Mar,1=Sharad"

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
| `--tracks`       | off              | `"0=Mar,1=Sharad"` — label by audio track (exact, no diarization) |
| `--device`       | `cpu`            | `cuda` is **NVIDIA-only**                          |
| `--language`     | autodetect       | force a code like `en`/`ko` to skip detection      |
| `-f/--format`    | `txt`            | repeatable: `txt`, `srt`, `vtt`, `json`            |
| `--no-punctuate` | off              | skip ML sentence/punctuation restoration           |
| `--no-tidy`      | off              | keep raw casing/spacing (skip light readability pass) |

First run downloads the model (large-v3 ≈ 3 GB) to `~/.cache/huggingface`.

### Readability / punctuation

Whisper emits long, lightly-punctuated runs on casual speech. With the `readable`
extra installed, each transcript is re-punctuated and split into sentences over
the full per-speaker stream. It's a small truecasing model, so expect occasional
over-capitalization (e.g. a word after a comma) — the trade for far more readable
text. Use `--no-punctuate` for verbatim, unpunctuated output.

## Multi-track recordings — the most accurate speaker labels

If your recorder captured each person on a **separate audio track** — e.g. AMD
ReLive's *Separate Microphone Track* (your mic on one track, the meeting app's
audio on another) — skip acoustic diarization entirely:

```pwsh
uv run video-transcribe meeting.mp4 --list-tracks            # find the indices
uv run video-transcribe meeting.mp4 --tracks "0=Mar,1=Sharad"
```

Each track is transcribed independently, labeled by who's on it, then the streams
are merged by timestamp. This gives **exact** speaker attribution (no guessing),
is faster (no pyannote, no word timestamps), and sidesteps alignment artifacts.
Record with **headphones** so your mic doesn't pick up the other side (bleed).

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
  audio.py        # ffmpeg: probe duration/streams, extract 16 kHz mono WAV (per-track)
  transcribe.py   # faster-whisper wrapper -> Segment / Word / TranscriptionResult
  diarize.py      # pyannote wrapper -> SpeakerTurn (in-memory WAV, no torchcodec)
  merge.py        # assign speakers (diarization OR per-track), clean, regroup
  punctuate.py    # optional sentence/punctuation restoration (punctuators/ONNX)
  correct.py      # apply a glossary (term fixes + speaker names) to a transcript
  formats.py      # readable txt + srt / vtt / json writers
  cli.py          # argument parsing + orchestration (diarize / --tracks modes)
tests/
  smoke.py        # model-free merge/format checks
  real_check.py   # faster-whisper word-timestamps -> merge (no HF token)
```
