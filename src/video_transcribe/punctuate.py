"""Optional punctuation / capitalisation / sentence-segmentation restoration.

Whisper emits long, lightly-punctuated runs on casual speech. `punctuators`
(ONNX, via the already-present onnxruntime) restores sentence boundaries,
punctuation and capitalisation -- the single biggest readability win.

Applied per ASR *segment* (each is short), so the model's max sequence length is
never exceeded and no text is dropped.
"""

from __future__ import annotations

import importlib.util
import re

DEFAULT_PUNCT_MODEL = "pcs_en"

# The model occasionally emits a literal <unk> token (e.g. for "mm-hmm"); strip it.
_UNK = re.compile(r"<\s*unk\s*>", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
# Punctuation the model re-adds itself; stripped from input (apostrophes/hyphens kept).
_MODEL_PUNCT = re.compile(r"[.,!?;:…\"]")


def _normalize_in(text: str) -> str:
    """Prepare text for pcs_en, which expects lowercase, unpunctuated input.

    Capital letters get tokenized as <unk> (dropping the leading letter), so we
    lowercase and strip model-added punctuation; the model then re-cases and
    re-punctuates from a clean slate.
    """
    return re.sub(r"\s+", " ", _MODEL_PUNCT.sub(" ", text)).strip().lower()


def available() -> bool:
    return importlib.util.find_spec("punctuators") is not None


def load(model_name: str = DEFAULT_PUNCT_MODEL):
    from punctuators.models import PunctCapSegModelONNX
    return PunctCapSegModelONNX.from_pretrained(model_name)


def _clean(text: str) -> str:
    text = _UNK.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


def restore(model, texts: list[str], *, batch_size: int = 32) -> list[str]:
    """Re-punctuate each input; its sentences are rejoined with a space.

    Output is aligned 1:1 with `texts`; empty inputs pass through unchanged.
    """
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        normalized = [(j, _normalize_in(t)) for j, t in enumerate(chunk)]
        keep = [(j, n) for j, n in normalized if n]
        results = model.infer([n for _, n in keep]) if keep else []
        restored = {
            j: _clean(" ".join(s.strip() for s in sents if s.strip()))
            for (j, _), sents in zip(keep, results)
        }
        out.extend(restored.get(j, chunk[j]) for j in range(len(chunk)))
    return out
