"""Speaker diarization (who-spoke-when) via pyannote.audio.

Two Windows-specific notes baked in here:

* pyannote 4.x decodes audio through `torchcodec`, whose native DLLs are flaky
  on Windows. We dodge that entirely by reading the (known 16 kHz mono PCM) WAV
  with the stdlib `wave` module and handing pyannote an in-memory waveform.
* The diarization model is gated on Hugging Face, so it needs a free token plus
  one-time acceptance of the model terms; `DiarizationError` explains how.

`load_pipeline` is split from `run_pipeline` so a caller can validate the token
up front (fast) before spending minutes on transcription, and reuse one loaded
pipeline across many files.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str  # raw pyannote label, e.g. "SPEAKER_00"


class DiarizationError(RuntimeError):
    """Raised when the diarization model can't be loaded or run."""


def _load_waveform(wav_path: Path):
    """Read a 16-bit PCM WAV into a (channels, time) float32 torch tensor."""
    import numpy as np
    import torch

    with wave.open(str(wav_path), "rb") as w:
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sample_rate = w.getframerate()
        raw = w.readframes(w.getnframes())

    if sample_width != 2:
        raise DiarizationError(f"expected 16-bit PCM WAV, got {sample_width * 8}-bit")

    data = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).T  # (channels, time)
    else:
        data = data.reshape(1, -1)
    return torch.from_numpy(data.copy()), sample_rate


def _token_help(model: str, err: object) -> str:
    return (
        f"Could not load diarization model '{model}'.\n"
        f"  reason: {err}\n\n"
        "Speaker diarization needs a (free) Hugging Face token AND one-time\n"
        "acceptance of the gated model terms. Log in with `uv run hf auth login`\n"
        "(token from https://huggingface.co/settings/tokens, read scope), then\n"
        "click 'Agree and access' on:\n"
        f"  - https://huggingface.co/{model}\n"
        "If that repo references other gated repos, accept those too, then re-run.\n"
    )


def load_pipeline(model: str = DEFAULT_DIARIZE_MODEL, *, hf_token: str | None,
                  num_threads: int | None = None):
    """Load (and authenticate) the pyannote pipeline on CPU. Fails fast/clearly."""
    import torch
    from pyannote.audio import Pipeline

    if num_threads:
        torch.set_num_threads(num_threads)

    # token=None lets huggingface_hub fall back to a cached `hf auth login` token,
    # so the caller can authenticate locally without passing the secret around.
    try:
        pipeline = Pipeline.from_pretrained(model, token=hf_token)
    except Exception as e:  # network / auth / gating / version issues
        raise DiarizationError(_token_help(model, e)) from e
    if pipeline is None:
        # pyannote returns None (instead of raising) when terms aren't accepted
        # or no usable token is available.
        raise DiarizationError(_token_help(model, "not authenticated, or model terms not accepted"))

    pipeline.to(torch.device("cpu"))
    return pipeline


def run_pipeline(
    pipeline,
    wav_path: Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerTurn]:
    """Run a loaded pipeline over a WAV and return sorted speaker turns."""
    waveform, sample_rate = _load_waveform(Path(wav_path))

    kwargs: dict[str, int] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    output = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)

    # pyannote 4.x returns a DiarizeOutput wrapper; older releases (and some
    # pipelines) return the Annotation directly.
    annotation = getattr(output, "speaker_diarization", output)

    turns = [
        SpeakerTurn(start=float(seg.start), end=float(seg.end), speaker=str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: (t.start, t.end))
    return turns


def diarize(
    wav_path: Path,
    *,
    hf_token: str | None,
    model: str = DEFAULT_DIARIZE_MODEL,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    num_threads: int | None = None,
    on_status=None,
) -> list[SpeakerTurn]:
    """Convenience: load + run in one call."""
    if on_status:
        on_status(f"loading diarization model '{model}' ...")
    pipeline = load_pipeline(model, hf_token=hf_token, num_threads=num_threads)
    if on_status:
        on_status("running diarization (this is the slow part on CPU) ...")
    return run_pipeline(pipeline, wav_path, num_speakers=num_speakers,
                        min_speakers=min_speakers, max_speakers=max_speakers)
