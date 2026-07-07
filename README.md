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
uv sync --extra tui                        # + the terminal UI (see below)
```

- `tui` pulls in Textual + psutil (tiny); adds a terminal UI that lists every
  command and option — configure, run, and monitor jobs without memorizing flags.
- `diarize` pulls in torch + pyannote (~2–3 GB); needed for `--diarize`.
- `readable` pulls in `punctuators` (tiny — reuses the onnxruntime faster-whisper
  already installs); enables sentence/punctuation restoration. On by default when
  installed; disable per-run with `--no-punctuate`.
- `llm` pulls in the `anthropic` SDK; needed for the optional
  `llm_correct.py` correction pass (sends transcript text to the Claude API —
  see below). Not needed for anything else in this tool.

## Terminal UI

If you'd rather not memorize flags, the TUI puts every command and option in one
place — pick a task, fill in the form, watch the exact command build live, then
run and monitor it (several at once, if you like):

```pwsh
uv sync --extra tui          # one-time
uv run video-transcribe-tui  # launch
```

- **Left:** every task, grouped and ordered by how often you'd run them —
  Transcribe · Transcribe + diarize · List audio tracks · by-track modes ·
  Correct (glossary / LLM) · Voiceprints (list / enroll / validate) · Mux ·
  Environment doctor. Type to fuzzy-filter.
- **Right:** ready-made **examples** for each task (the common runs from this
  project — ReLive separate-mic, diarize, glossary/LLM correction, voiceprints);
  click **Load** to drop one into the form. Below them, one widget per CLI option
  (files, dropdowns, switches), a live `$ python -m video_transcribe …` preview of
  the command, and **Run** / **Stop**.
- **Bottom:** a live progress line (Whisper/pyannote/download bars animate in
  place) over a full, colour-highlighted scrollback log. `w` saves the log,
  `y` copies it, `c` clears it.

New here? Run the **Environment doctor** task first — it checks ffmpeg, the
optional extras, and the diarization / LLM tokens, and tells you exactly what to
install if something's missing (also available standalone:
`uv run python -m video_transcribe.tui_doctor`).

Every task in the TUI maps 1:1 to the CLI below — it just builds the same
command for you, so anything here is still scriptable by hand.

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

If each person was captured on a **separate audio source** — e.g. AMD ReLive's
*Separate Microphone Track* (your mic vs the meeting app's audio) — skip acoustic
diarization entirely and label by track. Two layouts:

**Two streams in one file** (mic muxed into the video):
```pwsh
uv run video-transcribe meeting.mp4 --list-tracks            # find the indices
uv run video-transcribe meeting.mp4 --tracks "0=Mar,1=Sharad"
```

**Two separate files** (mic written to its own file, e.g. `.m4a`):
```pwsh
uv run video-transcribe meeting.mp4 meeting.m4a --track-speakers "Mar,Sharad"
```

Each track is transcribed independently and merged by timestamp — **exact** speaker
attribution (no guessing), faster (no pyannote/word-timestamps), no alignment
artifacts. Record with **headphones** so your mic doesn't pick up the other side.

**Group call + separate mic** (the other side is *several* people mixed into one
track, but you were captured on your own mic): diarize just that one input and
label the rest by track.
```pwsh
uv run video-transcribe meeting.mp4 meeting.m4a \
  --diarize-track 0 --speakers 4 --track-speakers "Sharad"
```
`--diarize-track 0` acoustically diarizes input file 0's audio (up to
`--speakers`/`--min-speakers`/`--max-speakers` people); `--track-speakers` names
the *other* input file(s), one name per remaining file, in order. Diarized
speakers come out generic (`Speaker 1`, `Speaker 2`, ...) — match voices to names
and rename with `correct.py` afterwards, or auto-name them by voice (see
Voiceprints below). `--mux` works the same way here too.

### Merge video + separate mic into one playable file

Standard players play only one audio track at a time, so to get a file where
**both** play on hit-play *and* the mic stays isolated, add a default **Mix** track
beside the originals (titles are the generic `Mix` / `Desktop` / `Mic`):

```pwsh
# all-in-one: transcript + a .mkv with Mix(default) + Desktop + Mic tracks
uv run video-transcribe meeting.mp4 meeting.m4a --track-speakers "Mar,Sharad" --mux

# just merge, no transcript:
uv run python -m video_transcribe.mux meeting.mp4 meeting.m4a   # -> meeting.with-mic.mkv
```

The video is stream-copied (no re-encode); only the small Mix track is encoded.

## Voiceprints (optional, auto-identify diarized speakers by voice)

Diarization always starts out generic (`Speaker 1`, `Speaker 2`, ...) — matching
voices to names by listening/reading is a one-time cost per *person*, not per
meeting. `voiceprint.py` makes that permanent: enroll a few confirmed segments
per person once, and future diarized recordings get auto-labeled with their
real name instead of a generic placeholder.

Fully local — embeddings never leave the machine, and it reuses the embedding
model already bundled inside the diarization pipeline (no extra Hugging Face
gating beyond what `--diarize` already needs).

**Enroll** from a transcript you've already confirmed is correctly named (e.g.
after `correct.py`), pointing at whichever audio/video file that speech
actually came from:

```pwsh
# from a plain diarized recording:
uv run python -m video_transcribe.voiceprint enroll meeting.json meeting.mp4 \
  --store voiceprints.json

# from a hybrid video+mic transcript, restrict to the names that actually live
# in *that* file (--list-tracks on the main CLI shows track indices):
uv run video-transcribe meeting.with-mic.mkv --list-tracks
uv run python -m video_transcribe.voiceprint enroll meeting.json meeting.with-mic.mkv \
  --store voiceprints.json --names "Ryan,Mar,Ness,John" --track 1   # Desktop
uv run python -m video_transcribe.voiceprint enroll meeting.json meeting.with-mic.mkv \
  --store voiceprints.json --names "Sharad" --track 2               # Mic

uv run python -m video_transcribe.voiceprint list --store voiceprints.json
```

**Identify** on future recordings by passing the store to the main CLI —
diarized speakers that confidently match a known voice come out named
directly; anyone else still falls back to generic `Speaker N`:

```pwsh
uv run video-transcribe meeting.mp4 --diarize --speakers 4 --voiceprints voiceprints.json
```

Assignment is **one-to-one**: each enrolled person can win at most one
diarized speaker per recording, greedily by best score. This matters — the
person with the most enrollment samples otherwise tends to out-score the true
speaker on several clusters at once (measured on a ground-truth meeting: 3/5
speakers correct with independent matching vs 5/5 with exclusive assignment).
`--voice-threshold` (default `0.5`, cosine similarity) mainly guards against
matching people who aren't enrolled at all — raise it if an unknown voice
gets a name it shouldn't, lower it if an obviously correct match is being
left generic. There's a `validate` subcommand to check the store against a
transcript whose names you've already confirmed, and
`tests/eval_matching.py` to compare scoring variants offline:

```pwsh
uv run python -m video_transcribe.voiceprint validate meeting.json meeting.mp4 --store voiceprints.json
```

This is **self-learning** by habit, not automatically: every time you confirm
a `correct.py` mapping is right, enroll that transcript too, and accuracy
compounds the more the tool is used.

The store is just a JSON file of embedding vectors per name — small, but it
*is* biometric-ish data about named real people, so keep it out of version
control and anywhere you wouldn't put a phone book with voice tags (a location
outside any git repo works well).

## LLM-assisted correction (optional, sends text to Claude)

`correct.py` fixes known terms via a fixed glossary (deterministic, fully
local). For everything else — misheard names/jargon not yet in the glossary,
odd ASR homophone slips — `llm_correct.py` runs those utterances past the
Claude API as a **correction-only** pass: it's instructed to fix mishearings
and obvious ASR errors and nothing else (no rewriting, summarizing, merging,
or inventing content), and it must return one corrected string per input
utterance so the result stays 1:1 with the original.

```pwsh
uv sync --extra llm                          # installs the anthropic SDK
uv run python -m video_transcribe.llm_correct meeting.json --glossary g.json
```

Needs `ANTHROPIC_API_KEY` (or `ant auth login`). Output goes to **separate**
files next to the input, so nothing is overwritten and corrections are easy to
review:

- `meeting.llm.txt` / `meeting.llm.json` — the corrected transcript
- `meeting.llm-changes.txt` — only the utterances that changed, old → new

Only utterance-level text (the paragraph grain in `.txt`/`.json`) is
corrected; segments (the subtitle-cue grain used for `.srt`/`.vtt`) aren't
touched, so this tool only writes `txt`/`json`. Costs roughly a few cents per
meeting on the default model (`claude-opus-4-8`; override with `--model`).

**Privacy note:** unlike the rest of this pipeline (fully local), this sends
the transcript text — including names and meeting content — to Anthropic's
API. Skip it for anything that shouldn't leave the machine.

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
  audio.py        # ffmpeg: probe, extract per-track WAV, mux video+mic
  transcribe.py   # faster-whisper wrapper -> Segment / Word / TranscriptionResult
  diarize.py      # pyannote wrapper -> SpeakerTurn (in-memory WAV, no torchcodec)
  merge.py        # assign speakers (diarization OR per-track), clean, regroup
  punctuate.py    # optional sentence/punctuation restoration (punctuators/ONNX)
  correct.py      # apply a glossary (term fixes + speaker names) to a transcript
  llm_correct.py  # optional Claude-API correction pass (separate .llm.* output)
  mux.py          # merge video + separate mic -> one MKV (Mix/Desktop/Mic)
  formats.py      # readable txt + srt / vtt / json writers
  cli.py          # argument parsing + orchestration (diarize / tracks / mux)
  tui.py          # Textual terminal UI (video-transcribe-tui)
  tui_catalog.py  # data-only task/option catalog -> subprocess argv (no UI deps)
  tui_stream.py   # split subprocess output into live progress vs scrollback log
  tui_highlight.py# Rich log highlighter tuned to this tool's output
  tui_doctor.py   # environment/setup checks (also `python -m ...tui_doctor`)
tests/
  smoke.py        # model-free merge/format checks
  real_check.py   # faster-whisper word-timestamps -> merge (no HF token)
  tui_check.py    # catalog argv-building, stream demux, doctor, Textual pilot
```
