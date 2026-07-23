"""Persistent voice embeddings for auto-identifying diarized speakers by voice,
instead of leaving them as generic "Speaker N" every time.

A voiceprint is just one or more speaker-embedding vectors (from pyannote's
embedding model) for a known person, stored in a small local JSON file --
nothing ever leaves the machine. `identify_turns` matches a new recording's
diarized clusters against the store by cosine similarity; `enroll_from_transcript`
grows the store from a transcript you've already confirmed is correctly named
(e.g. after `correct.py`), so accuracy compounds the more you use this tool.

Usage:
  # bootstrap/grow the store from an already-corrected transcript + its source
  # audio/video (only speakers whose audio actually lives in that file):
  uv run python -m video_transcribe.voiceprint enroll TRANSCRIPT.json MEDIA.mp4 \\
      --store voiceprints.json --names "Ryan,Mar,Ness,John"

  uv run python -m video_transcribe.voiceprint list --store voiceprints.json

Then pass `--voiceprints voiceprints.json` to the main `video-transcribe` CLI to
auto-name diarized speakers on future recordings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from video_transcribe.diarize import SpeakerTurn, load_waveform

# Embedding models are unstable over very short clips; skip anything shorter.
MIN_SEGMENT_SECONDS = 1.5
# Chosen empirically on a ground-truth meeting whose identification audio was
# re-encoded (worst case measured so far): true cluster-level matches scored
# 0.55-0.90 while impostor scores stayed below ~0.67 -- and the exclusive
# one-to-one assignment in identify_turns handles enrolled-speaker confusions,
# so this threshold's main job is rejecting people who aren't enrolled at all.
# The old 0.75 default rejected correct matches on channel-mismatched audio.
DEFAULT_MATCH_THRESHOLD = 0.5
# Reuse the embedding sub-model bundled inside the diarization pipeline's own
# repo (loaded via `subfolder=`) rather than the separately-gated top-level
# `pyannote/embedding` repo -- if diarization already works, this needs no
# extra Hugging Face access grant.
DEFAULT_EMBEDDING_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_EMBEDDING_SUBFOLDER = "embedding"


class VoiceprintError(RuntimeError):
    """Raised when the embedding model can't be loaded."""


def load_embedder(
    model: str = DEFAULT_EMBEDDING_MODEL, *,
    subfolder: str | None = DEFAULT_EMBEDDING_SUBFOLDER, hf_token: str | None = None,
):
    """Load a pyannote embedding model as a whole-clip `Inference` callable."""
    import torch
    from pyannote.audio import Inference, Model

    try:
        pt_model = Model.from_pretrained(model, subfolder=subfolder, token=hf_token)
    except Exception as e:  # network / auth / gating / version issues
        raise VoiceprintError(
            f"Could not load embedding model '{model}' (subfolder={subfolder!r}).\n"
            f"  reason: {e}\n\n"
            "This needs the same kind of (free) Hugging Face token/access as "
            f"diarization -- log in with `uv run hf auth login` and accept the "
            f"model terms at https://huggingface.co/{model}\n"
        ) from e
    if pt_model is None:
        raise VoiceprintError(
            f"Could not load embedding model '{model}' "
            "(not authenticated, or model terms not accepted)."
        )
    inference = Inference(pt_model, window="whole")
    inference.to(torch.device("cpu"))
    return inference


def embed_waveform(embedder, waveform, sample_rate: int, start: float, end: float):
    """Embed the [start, end) crop of an already-loaded waveform tensor.

    Returns None if the crop is too short to embed reliably.
    """
    import numpy as np

    i0, i1 = int(max(0.0, start) * sample_rate), int(max(0.0, end) * sample_rate)
    if i1 - i0 < int(MIN_SEGMENT_SECONDS * sample_rate):
        return None
    crop = waveform[:, i0:i1]
    vec = embedder({"waveform": crop, "sample_rate": sample_rate})
    return np.asarray(vec).reshape(-1)


@dataclass
class VoiceprintStore:
    """Known people -> their enrolled embedding vectors, persisted as JSON."""

    path: Path
    people: dict[str, list[list[float]]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "VoiceprintStore":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(path=path, people=data.get("people", {}))
        return cls(path=path, people={})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"people": self.people}, indent=2), encoding="utf-8",
        )

    def add(self, name: str, embedding) -> None:
        self.people.setdefault(name, []).append([float(x) for x in embedding])

    def centroid(self, name: str):
        # Length-normalize each sample BEFORE averaging (verified best
        # practice from the multi-enrollment speaker-verification literature,
        # e.g. Rajan et al., Digital Signal Processing 2014) so one loud/long
        # sample can't dominate the mean direction.
        import numpy as np
        samples = np.array(self.people[name])
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        return np.mean(samples, axis=0)

    def scores(self, embedding) -> dict[str, float]:
        """Cosine similarity of `embedding` against every enrolled centroid."""
        import numpy as np

        emb = np.asarray(embedding, dtype=float)
        emb = emb / np.linalg.norm(emb)
        out: dict[str, float] = {}
        for name in self.people:
            c = self.centroid(name)
            out[name] = float(np.dot(emb, c / np.linalg.norm(c)))
        return out

    def match(
        self, embedding, *, threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> tuple[str | None, float]:
        """Best-matching known name for `embedding`, or (None, best_score) if
        nothing clears `threshold`.

        Note: this is an *independent* 1:N match; when assigning several
        diarized clusters from one recording, prefer the exclusive one-to-one
        assignment in `identify_turns`, which prevents one enrolled voice from
        claiming multiple speakers.
        """
        if not self.people:
            return None, 0.0
        by_name = self.scores(embedding)
        best_name = max(by_name, key=by_name.get)
        best_score = by_name[best_name]
        if best_score < threshold:
            return None, best_score
        return best_name, best_score


def identify_turns(
    turns: list[SpeakerTurn], waveform, sample_rate: int, embedder, store: VoiceprintStore,
    *, threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict[str, str]:
    """Match raw diarized speaker labels to known voiceprint names, one-to-one.

    Assignment is *exclusive*: each enrolled person can win at most one
    diarized cluster, greedily by descending score. Without this, the person
    with the most/tightest enrollment data tends to out-score the true
    speaker on several clusters at once (measured: 3/5 clusters correct
    independently vs 5/5 with exclusive assignment on the same meeting).
    One diarized label per real person is exactly the diarizer's own contract,
    so exclusivity encodes a constraint the independent argmax throws away.

    Returns {raw_label: matched_name} for confident matches only -- callers
    should fall back to the generic 'Speaker N' label for any label not here.
    """
    import numpy as np

    by_label: dict[str, list[SpeakerTurn]] = defaultdict(list)
    for t in turns:
        by_label[t.speaker].append(t)

    label_scores: dict[str, dict[str, float]] = {}
    for label, label_turns in by_label.items():
        vecs = [
            v for t in label_turns
            if (v := embed_waveform(embedder, waveform, sample_rate, t.start, t.end)) is not None
        ]
        if not vecs:
            continue
        label_scores[label] = store.scores(np.mean(vecs, axis=0))

    ranked = sorted(
        ((score, label, name)
         for label, by_name in label_scores.items()
         for name, score in by_name.items()),
        key=lambda x: -x[0],
    )
    result: dict[str, str] = {}
    taken: set[str] = set()
    for score, label, name in ranked:
        if score < threshold:
            break  # ranked descending -- everything after is below threshold too
        if label in result or name in taken:
            continue
        result[label] = name
        taken.add(name)
    return result


def _iter_utterance_embeddings(
    transcript_path: Path, media_path: Path, embedder, *,
    min_seconds: float = MIN_SEGMENT_SECONDS, stream_index: int | None = None,
):
    """Yield (utterance_dict, embedding) for each utterance in a transcript
    long enough to embed, sourcing audio from `media_path`.

    `stream_index` (0-based, among audio streams) picks one audio track out of
    a multi-track file, e.g. a muxed Mix/Desktop/Mic recording -- use the
    Desktop or Mic track for a cleaner signal rather than the Mix.
    """
    from video_transcribe import audio

    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    utterances = data.get("utterances", [])

    with tempfile.TemporaryDirectory(prefix="video-transcribe-voiceprint-") as tmp:
        wav = Path(tmp) / (media_path.stem + ".16k.wav")
        audio.extract_audio(media_path, wav, stream_index=stream_index)
        waveform, sample_rate = load_waveform(wav)

        for u in utterances:
            if u["end"] - u["start"] < min_seconds:
                continue
            vec = embed_waveform(embedder, waveform, sample_rate, u["start"], u["end"])
            if vec is not None:
                yield u, vec


def enroll_from_transcript(
    transcript_path: Path, media_path: Path, store: VoiceprintStore, embedder,
    *, names: set[str] | None = None, min_seconds: float = MIN_SEGMENT_SECONDS,
    stream_index: int | None = None,
) -> dict[str, int]:
    """Add voiceprints from a transcript's utterances, sourcing audio from `media_path`.

    `names` restricts enrollment to those speakers (default: all) -- needed for
    e.g. a hybrid video+mic transcript where only some names' audio actually
    lives in this particular file.

    Returns {name: n_segments_added}.
    """
    added: dict[str, int] = {}
    for u, vec in _iter_utterance_embeddings(
        transcript_path, media_path, embedder, min_seconds=min_seconds, stream_index=stream_index,
    ):
        name = u.get("speaker")
        if not name or (names is not None and name not in names):
            continue
        store.add(name, vec)
        added[name] = added.get(name, 0) + 1
    return added


def validate_against_transcript(
    transcript_path: Path, media_path: Path, store: VoiceprintStore, embedder,
    *, threshold: float = DEFAULT_MATCH_THRESHOLD, min_seconds: float = MIN_SEGMENT_SECONDS,
    stream_index: int | None = None, names: set[str] | None = None,
) -> dict[str, list[dict]]:
    """Check the store's matches against a transcript's *already-confirmed*
    speaker labels -- a sanity check before trusting the store, or before
    growing it further from this same recording.

    Utterances for a speaker not yet in the store are skipped (nothing to
    validate against), as are speakers outside `names` when given -- needed
    for multi-file recordings where only some speakers' audio lives in
    `media_path` (e.g. video+separate-mic). Returns {"correct": [...],
    "wrong": [...], "no_match": [...]} of {"speaker", "matched", "score",
    "start", "text", "embedding"} dicts -- "embedding" lets a caller
    selectively enroll confirmed-correct, high-confidence rows without
    re-extracting audio (see bolster_correct).
    """
    results: dict[str, list[dict]] = {"correct": [], "wrong": [], "no_match": []}
    for u, vec in _iter_utterance_embeddings(
        transcript_path, media_path, embedder, min_seconds=min_seconds, stream_index=stream_index,
    ):
        true_name = u.get("speaker")
        if not true_name or true_name not in store.people:
            continue
        if names is not None and true_name not in names:
            continue
        matched, score = store.match(vec, threshold=threshold)
        row = {"speaker": true_name, "matched": matched, "score": round(score, 3),
              "start": u["start"], "text": u["text"][:60], "embedding": vec}
        if matched is None:
            results["no_match"].append(row)
        elif matched == true_name:
            results["correct"].append(row)
        else:
            results["wrong"].append(row)
    return results


def bolster_correct(
    store: VoiceprintStore, correct_rows: list[dict], *,
    min_score: float = 0.8, names: set[str] | None = None,
) -> dict[str, int]:
    """Enroll a subset of validate_against_transcript's "correct" rows --
    ones the store already matched correctly *and* confidently -- rather
    than blindly re-enrolling everything from a recording that only
    partially validated. Returns {name: n_added}.
    """
    added: dict[str, int] = {}
    for row in correct_rows:
        if row["score"] < min_score:
            continue
        if names is not None and row["speaker"] not in names:
            continue
        store.add(row["speaker"], row["embedding"])
        added[row["speaker"]] = added.get(row["speaker"], 0) + 1
    return added


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="video-transcribe-voiceprint",
        description="Enroll/inspect voiceprints for auto speaker identification.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    enroll_p = sub.add_parser(
        "enroll", help="Add voiceprints from an already-corrected transcript + its source audio/video.",
    )
    enroll_p.add_argument("transcript", type=Path, help="corrected transcript .json (real speaker names)")
    enroll_p.add_argument("media", type=Path, help="the audio/video file that transcript's speech came from")
    enroll_p.add_argument("--store", type=Path, required=True, help="voiceprint store JSON")
    enroll_p.add_argument("--names", default=None,
                          help="Comma-separated names to enroll from this file "
                               "(default: all speakers in the transcript)")
    enroll_p.add_argument("--track", type=int, default=None, metavar="IDX",
                          help="0-based audio track index for a multi-track file "
                               "(e.g. a muxed Mix/Desktop/Mic .mkv -- pick Desktop or "
                               "Mic for a cleaner voiceprint than the Mix). See "
                               "--list-tracks in the main CLI to find indices.")
    enroll_p.add_argument("--hf-token", default=None)

    validate_p = sub.add_parser(
        "validate", help="Check the store's matches against a transcript's already-confirmed "
                         "speaker labels, without changing the store.",
    )
    validate_p.add_argument("transcript", type=Path, help="corrected transcript .json (real speaker names)")
    validate_p.add_argument("media", type=Path, help="the audio/video file that transcript's speech came from")
    validate_p.add_argument("--store", type=Path, required=True, help="voiceprint store JSON")
    validate_p.add_argument("--names", default=None,
                            help="Comma-separated speakers to validate from this file "
                                 "(default: all in the transcript that are enrolled)")
    validate_p.add_argument("--track", type=int, default=None, metavar="IDX")
    validate_p.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    validate_p.add_argument("--hf-token", default=None)

    list_p = sub.add_parser("list", help="List known people and how many samples each has.")
    list_p.add_argument("--store", type=Path, required=True)

    args = p.parse_args(argv)

    if args.command == "list":
        store = VoiceprintStore.load(args.store)
        if not store.people:
            print("no voiceprints enrolled yet")
            return 0
        for name, vecs in sorted(store.people.items()):
            print(f"{name}: {len(vecs)} sample(s)")
        return 0

    if args.command == "validate":
        store = VoiceprintStore.load(args.store)
        v_names = ({n.strip() for n in args.names.split(",") if n.strip()}
                   if args.names else None)
        try:
            print(f"loading embedding model '{DEFAULT_EMBEDDING_MODEL}' ...", file=sys.stderr)
            embedder = load_embedder(hf_token=args.hf_token or os.environ.get("HF_TOKEN"))
            results = validate_against_transcript(
                args.transcript, args.media, store, embedder,
                threshold=args.threshold, stream_index=args.track, names=v_names,
            )
        except VoiceprintError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        total = sum(len(v) for v in results.values())
        if total == 0:
            print("no comparable utterances (no speaker in this transcript is in the store yet)",
                  file=sys.stderr)
            return 1
        n_correct = len(results["correct"])
        n_wrong = len(results["wrong"])
        for row in results["wrong"]:
            print(f"  WRONG true={row['speaker']:<8} matched={row['matched']:<8} "
                  f"score={row['score']} [{row['start']:.0f}s] {row['text']}")
        for row in results["no_match"]:
            print(f"  MISS  true={row['speaker']:<8} (below threshold, score={row['score']}) "
                  f"[{row['start']:.0f}s] {row['text']}")
        print(f"{n_correct}/{total} correct ({n_correct / total * 100:.0f}%)")
        # Exit 0 iff there are zero WRONG matches -- a miss is harmless (the
        # utterance just isn't enrolled), but a wrong match teaches the store
        # a lie. Callers automating "validate, then enroll if it passed"
        # should gate on this exit code, not on 100% (misses are routine on
        # short utterances and don't make enrollment unsafe).
        return 0 if n_wrong == 0 else 2

    # enroll
    store = VoiceprintStore.load(args.store)
    names = {n.strip() for n in args.names.split(",") if n.strip()} if args.names else None
    try:
        print(f"loading embedding model '{DEFAULT_EMBEDDING_MODEL}' ...", file=sys.stderr)
        embedder = load_embedder(hf_token=args.hf_token or os.environ.get("HF_TOKEN"))
        added = enroll_from_transcript(args.transcript, args.media, store, embedder,
                                       names=names, stream_index=args.track)
    except VoiceprintError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not added:
        print("no matching speaker segments found (check --names / the transcript's speaker labels)",
              file=sys.stderr)
        return 1
    store.save()
    for name, n in sorted(added.items()):
        print(f"enrolled {n} segment(s) for {name}")
    print(f"wrote {args.store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
