"""Visual speaker identification from Google Meet gallery-view recordings.

Audio-based diarization (pyannote, see diarize.py/voiceprint.py) can silently
MERGE two different real speakers into one cluster when their utterances are
short or acoustically similar -- this happened for real (2026-08-12): a
14-minute meeting's diarized "Speaker 1" cluster actually contained two
different people, undetected until a human caught it by ear.

Google Meet's own gallery-view UI gives an independent, ground-truth-backed
signal for this: each participant tile has a name label, and the ACTIVE
SPEAKER gets a colored border around their tile. This module extracts video
frames at each utterance's timestamps, finds which tile has that border,
OCRs its name label, and uses the result for two things:

1. Auto-naming speakers still generic ("Speaker N") after diarization --
   composes with merge.py exactly like voiceprint.py's voice_names (a plain
   {raw_label: name} dict), so merge.py needed no changes for this.
2. The safety-critical piece this was built for: detecting when a single
   diarized label's utterances visually resolve to MORE THAN ONE distinct
   name -- i.e. reproducing, automatically, the kind of human catch that
   caught the real incident. This is only ever REPORTED, never auto-fixed --
   see check_label_consistency. A miss is harmless; a wrong silent guess is
   exactly what this module exists to prevent, so every rejection path here
   leaves a label exactly as safe as not running this at all.

Border color/geometry constants below were measured ad hoc against real
frames extracted from the incident recording during interactive development
of this module (no checked-in script preserves that measurement -- treat the
constants as [measured]-then-hardcoded, not derivable from a repo artifact),
not guessed from Meet's brand colors -- the on-screen border renders far less
saturated than Google's nominal blue, and a "vivid blue" assumption would have
rejected every real sample.

Fully local except for a one-time model download (EasyOCR's detection +
recognition weights, cached after first run) -- same precedent as
faster-whisper/pyannote's own model downloads elsewhere in this project.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Measured from the real incident recording (compressed screen-capture, dark
# Meet theme): border pixels clustered at hue 213-217 degrees, saturation only
# 0.18-0.34 (NOT a vivid/saturated blue), value 0.82-0.97. Generous margins
# either side of the measured range, not the raw range itself.
DEFAULT_BORDER_HUE_DEG = 215.0
DEFAULT_HUE_TOLERANCE_DEG = 18.0
DEFAULT_MIN_SATURATION = 0.10
DEFAULT_MAX_SATURATION = 0.65
DEFAULT_MIN_VALUE = 0.55

# Name-label crop, as a fraction of the detected tile bounding box: bottom
# band, left-aligned (avoids the mute-icon that sits bottom-right on most
# tiles). Empirically eyeballed from the one confirmed recording -- Meet's UI
# could shift this in a future release.
LABEL_TOP_FRAC = 0.82
LABEL_BOTTOM_MARGIN_PX = 4
LABEL_RIGHT_FRAC = 0.65

# A border is a thin PERIMETER stroke, not a filled region -- its pixel count
# scales with perimeter (~2*(h+w)*stroke_width), not area. A real measured
# example (337x417 tile, ~4px stroke) had area_frac=0.0026 (0.26%) -- an
# area-based floor here must stay well under that, not assume a filled blob.
MIN_TILE_AREA_FRAC = 0.0008
MAX_TILE_AREA_FRAC = 0.5
MIN_TILE_SIDE_PX = 20

# Perimeter-band width for the shape check, in PIXELS -- fixed, NOT scaled to
# the tile's bounding-box size. This was the second real bug found by testing:
# a band that scales with bbox size (bbox_side // 20) is way wider than the
# border stroke itself (measured ~3-4px regardless of tile size, since it's a
# fixed UI element), which dilutes perim_density for a genuine border down to
# the same ballpark as background noise and made real/fake indistinguishable
# by density alone. With a small FIXED band, ratio (interior/perimeter
# density) cleanly separates them: measured ~0.01-0.02 for a real border vs.
# ~0.17-0.30 for the false-positive background blob that motivated this whole
# rewrite, stable across band widths 3-10px. Picked 5px as a comfortable
# middle of that range.
BORDER_BAND_PX = 5
MAX_INTERIOR_TO_PERIMETER_RATIO = 0.10  # see measured gap above; real ~0.02, fake ~0.2+
MIN_PERIM_DENSITY = 0.30  # just rejects a near-empty band; NOT the primary discriminator
MAX_SIGNIFICANT_COMPETING_REGIONS = 2

DEFAULT_SAMPLE_SPACING = 2.0
MIN_SAMPLES_PER_UTTERANCE = 3
MAX_SAMPLES_PER_UTTERANCE = 8
SAMPLE_MARGIN_SECONDS = 0.3  # keep samples away from turn boundaries

# Floor on the gap between two sample timestamps within one utterance. Found
# by review (not yet observed live): for a short utterance, cramming in
# MIN_SAMPLES_PER_UTTERANCE=3 evenly-spaced points can land them only ~25ms
# apart -- often the identical ffmpeg-decoded video frame twice. That silently
# turns "2 independent accepted samples" (see MIN_MERGE_ACCEPTED_SAMPLES,
# check_label_consistency) into one piece of evidence double-counted as two.
# 0.4s is well above one frame duration even for a low-fps screen capture
# (~2.5fps); sample_points reduces the sample count for short utterances
# rather than violate this floor.
MIN_SAMPLE_INTERVAL_SECONDS = 0.4

DEFAULT_NAME_THRESHOLD = 0.6
ROSTER_AMBIGUITY_DELTA = 0.05  # top-2 fuzzy scores this close -> reject, don't guess

# A minority name is "supported" (see check_label_consistency) if EITHER it
# has this many utterances (any confidence) OR at least one utterance at/above
# MIN_MERGE_CONFIDENCE *with at least MIN_MERGE_ACCEPTED_SAMPLES agreeing
# samples*. The confidence path is what actually catches a genuine
# single-utterance second speaker (the real incident had exactly one); the
# count path additionally catches a persistent low-confidence pattern spread
# across utterances that no single reading alone would justify.
#
# The accepted-samples floor was added after testing against real video: a
# reading with exactly 1 accepted sample out of 8 attempted is trivially
# "confidence=1.0" (1 vote for the winner / 1 accepted sample), identical in
# that field to a real 7-of-8 unanimous reading, despite being far weaker
# evidence -- one stray misdetected frame, not a sustained visual match. Every
# false merge_suspected flag found by testing had exactly this shape (n=1/8,
# n=1/3, etc.); requiring >=2 agreeing samples filters those out while still
# accepting a real single utterance (which the real incident's own contributing
# reading had multiple agreeing samples for).
MIN_MERGE_SUPPORT_UTTERANCES = 2
MIN_MERGE_CONFIDENCE = 0.66
MIN_MERGE_ACCEPTED_SAMPLES = 2


class VisualIdError(RuntimeError):
    """Raised when a required visual-id dependency is missing or misused."""


@dataclass(frozen=True)
class BorderDetection:
    """A candidate active-speaker tile found in one frame."""
    bbox: tuple[int, int, int, int]  # (left, top, right, bottom), pixels
    confidence: float
    ambiguous_regions: int = 0  # >0 if other disjoint regions also matched


@dataclass(frozen=True)
class VisualReading:
    """The visual identification result for one utterance."""
    utterance_start: float
    utterance_end: float
    label: str | None
    name: str | None
    confidence: float          # winner votes / accepted samples
    coverage: float            # accepted samples / total samples
    n_samples: int
    n_accepted: int
    reject_reason: str | None  # set iff name is None
    candidates: dict[str, int] = field(default_factory=dict)  # accepted-name -> votes


@dataclass(frozen=True)
class MergeSuspected:
    """A diarized label whose visual readings resolved to >1 distinct name."""
    label: str
    names: dict[str, dict]  # name -> {"utterances": int, "duration": float}
    total_utterances: int


# --------------------------------------------------------------------------
# Pure geometry / matching logic -- no PIL/numpy/OCR imports at module scope,
# so importing this module (and running these functions against synthetic
# arrays or fake readers) never requires the `visual` extra. Mirrors
# voiceprint.py's convention of lazy-importing torch/numpy inside functions.
# --------------------------------------------------------------------------

def detect_border(
    frame,  # (H, W, 3) uint8 array-like
    *,
    target_hue: float = DEFAULT_BORDER_HUE_DEG,
    hue_tolerance: float = DEFAULT_HUE_TOLERANCE_DEG,
    min_saturation: float = DEFAULT_MIN_SATURATION,
    max_saturation: float = DEFAULT_MAX_SATURATION,
    min_value: float = DEFAULT_MIN_VALUE,
    min_perim_density: float = MIN_PERIM_DENSITY,
    max_significant_competitors: int = MAX_SIGNIFICANT_COMPETING_REGIONS,
) -> BorderDetection | None:
    """Find the active-speaker border in an RGB frame, or None if not found.

    Color-threshold (HSV) + a perimeter-vs-interior density shape check, pure
    NumPy -- no OpenCV. The shape check is what tells a thin border stroke
    apart from e.g. a participant wearing a blue shirt (a solid blob has high
    interior density too; a stroke doesn't).

    Two rejection gates found necessary by testing against the real incident
    recording, not from theory -- a hue/sat/val threshold alone matches all
    sorts of generic bluish-gray background/lighting/clothing content, not
    just the genuine border, and a naive "pick the biggest matching region"
    fallback silently locked onto that noise instead of returning None:

    - `min_perim_density`: a genuine stroke's bounding perimeter band should
      be MOSTLY the stroke color (near-solid once you account for
      anti-aliasing/compression). A real false positive found by this exact
      check had perim_density=0.39 -- scattered, sparse coverage across a
      big bounding box, not a stroke -- while still passing the old
      ratio-only check (interior was proportionally low too). Requiring the
      perimeter itself to be densely filled is a direct, much stronger gate.
    - `max_significant_competitors`: reject if more than this many OTHER
      regions are within 20% of the winning region's pixel count -- lots of
      similarly-sized matches means the threshold is picking up generic
      texture, not one specific highlighted tile. (Measured on the same false
      positive: winner 36484px, but 74 other regions matched, several of
      them >20% of the winner's size -- a genuine detection should have at
      most a couple of small, unrelated compression-noise specks.)
    """
    import numpy as np

    arr = np.asarray(frame, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    span = np.where((maxc - minc) == 0, 1.0, maxc - minc)
    s = np.where(maxc == 0, 0.0, (maxc - minc) / np.where(maxc == 0, 1.0, maxc))

    rc, gc, bc = (maxc - r) / span, (maxc - g) / span, (maxc - b) / span
    hue = np.zeros_like(v)
    hue = np.where(r == maxc, bc - gc, hue)
    hue = np.where(g == maxc, 2.0 + rc - bc, hue)
    hue = np.where(b == maxc, 4.0 + gc - rc, hue)
    hue_deg = (hue / 6.0 % 1.0) * 360.0

    hue_diff = np.minimum(np.abs(hue_deg - target_hue), 360.0 - np.abs(hue_deg - target_hue))
    mask = (
        (hue_diff <= hue_tolerance)
        & (s >= min_saturation) & (s <= max_saturation)
        & (v >= min_value)
    )

    h, w = mask.shape
    regions = _connected_regions(mask)
    if not regions:
        return None
    regions.sort(key=lambda reg: reg[1], reverse=True)  # by pixel count, descending

    # Evaluate every plausibly-sized candidate region by SHAPE, not just size --
    # a large unrelated blob (background/lighting/clothing) can easily outweigh
    # a genuine thin border stroke in raw pixel count. Bound to the top 20 by
    # size to cap cost; a real border has never been observed further down.
    # Collect every candidate that passes the SHAPE checks (not just the best
    # one) -- "significant competitors" below must only count other plausible
    # stroke-shaped regions, not just any large blob. Counting by raw size
    # regardless of shape was a real bug: a frame with lots of generic bluish
    # background content has several large regions that fail the shape check
    # entirely, and counting those as "competitors" made this reject almost
    # every genuine detection.
    passing: list[tuple[int, float]] = []  # (pixel count, confidence)
    best: BorderDetection | None = None
    best_count = 0
    best_idx = -1
    for (top, bottom, left, right), count in regions[:20]:
        bbox_h, bbox_w = bottom - top + 1, right - left + 1
        if bbox_h < MIN_TILE_SIDE_PX or bbox_w < MIN_TILE_SIDE_PX:
            continue
        area_frac = count / (h * w)
        if area_frac < MIN_TILE_AREA_FRAC or area_frac > MAX_TILE_AREA_FRAC:
            continue

        sub = mask[top:bottom + 1, left:right + 1]
        # Fixed width -- MIN_TILE_SIDE_PX (20) above guarantees bbox_h/bbox_w
        # >= 20, so min(bbox_h, bbox_w)//4 >= 5 always; a dynamic min() against
        # that was dead code (always resolved to BORDER_BAND_PX).
        band = BORDER_BAND_PX
        perim = np.zeros_like(sub)
        perim[:band, :] = True
        perim[-band:, :] = True
        perim[:, :band] = True
        perim[:, -band:] = True
        interior = ~perim

        perim_density = float(sub[perim].mean()) if perim.any() else 0.0
        if perim_density < min_perim_density:
            continue  # perimeter band isn't densely filled -- not a real stroke
        interior_density = float(sub[interior].mean()) if interior.any() else 0.0
        ratio = interior_density / max(perim_density, 1e-6)
        if ratio > MAX_INTERIOR_TO_PERIMETER_RATIO:
            continue  # looks like a filled blob, not a stroke

        confidence = max(0.0, min(1.0, (1.0 - ratio) * perim_density))
        passing.append((count, confidence))
        if best is None or confidence > best.confidence:
            best = BorderDetection(bbox=(left, top, right, bottom), confidence=confidence)
            best_count = count
            best_idx = len(passing) - 1

    if best is None:
        return None

    # Exclude the winner's own entry by INDEX, not by count equality -- a
    # distinct second region that happens to tie the winner's pixel count
    # exactly (plausible given _connected_regions' coarse block segmentation)
    # must still count as a competitor, not be silently excluded alongside
    # the winner itself.
    significant_competitors = sum(
        1 for idx, (count, _conf) in enumerate(passing)
        if idx != best_idx and 0.8 * best_count <= count <= 1.2 * best_count
    )
    if significant_competitors > max_significant_competitors:
        return None  # multiple similarly-sized stroke-shaped candidates -- ambiguous

    return BorderDetection(
        bbox=best.bbox, confidence=best.confidence, ambiguous_regions=significant_competitors,
    )


def _connected_regions(mask) -> list[tuple[tuple[int, int, int, int], int]]:
    """Coarse connected-component bounding boxes for a boolean mask.

    Deliberately simple (no scipy/opencv dependency): grid the mask into
    blocks, flood-fill over occupied blocks via a plain BFS, then take the
    pixel-accurate bbox of each block-cluster's actual mask pixels. Good
    enough to separate "the one bordered tile" from unrelated far-away noise;
    not meant to be pixel-perfect segmentation.
    """
    import numpy as np

    h, w = mask.shape
    block = max(4, min(h, w) // 100)
    bh, bw = (h + block - 1) // block, (w + block - 1) // block
    occupied = np.zeros((bh, bw), dtype=bool)
    for by in range(bh):
        for bx in range(bw):
            occupied[by, bx] = mask[by * block:(by + 1) * block, bx * block:(bx + 1) * block].any()

    visited = np.zeros_like(occupied)
    regions: list[tuple[tuple[int, int, int, int], int]] = []
    for by in range(bh):
        for bx in range(bw):
            if not occupied[by, bx] or visited[by, bx]:
                continue
            stack = [(by, bx)]
            visited[by, bx] = True
            cluster = []
            while stack:
                cy, cx = stack.pop()
                cluster.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < bh and 0 <= nx < bw and occupied[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            ys = [c[0] for c in cluster]
            xs = [c[1] for c in cluster]
            top, bottom = min(ys) * block, min((max(ys) + 1) * block, h) - 1
            left, right = min(xs) * block, min((max(xs) + 1) * block, w) - 1
            sub = mask[top:bottom + 1, left:right + 1]
            count = int(sub.sum())
            if count == 0:
                continue
            # tighten the bbox to the actual mask extent within this block-cluster
            rows = np.where(sub.any(axis=1))[0]
            cols = np.where(sub.any(axis=0))[0]
            regions.append((
                (top + int(rows.min()), top + int(rows.max()),
                 left + int(cols.min()), left + int(cols.max())),
                count,
            ))
    return regions


def label_crop_box(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Name-label sub-region within a detected tile bbox (left, top, right, bottom)."""
    left, top, right, bottom = bbox
    h, w = bottom - top + 1, right - left + 1
    label_top = top + int(h * LABEL_TOP_FRAC)
    label_bottom = max(label_top + 1, bottom - LABEL_BOTTOM_MARGIN_PX)
    label_right = left + int(w * LABEL_RIGHT_FRAC)
    return left, label_top, max(left + 1, label_right), label_bottom


def ocr_name(crop) -> str:
    """OCR a name-label crop ((H, W, 3) uint8 array). Requires the `visual` extra."""
    reader = _get_ocr_reader()
    import numpy as np

    arr = np.asarray(crop)
    if arr.size == 0 or arr.shape[0] < 4 or arr.shape[1] < 4:
        return ""
    # Upscale small crops -- EasyOCR (like most OCR) is unreliable below ~20px
    # text height, and a tile's name label crop is often smaller than that.
    scale = max(1, int(60 / max(arr.shape[0], 1)))
    if scale > 1:
        from PIL import Image
        img = Image.fromarray(arr).resize(
            (arr.shape[1] * scale, arr.shape[0] * scale), Image.LANCZOS,
        )
        arr = np.asarray(img)

    results = reader.readtext(arr)
    if not results:
        return ""
    # Concatenate left-to-right in case the name wrapped into >1 OCR box.
    results.sort(key=lambda r: r[0][0][0])
    return " ".join(text.strip() for _bbox, text, _conf in results if text.strip())


_OCR_READER = None


def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
        except ImportError as e:
            raise VisualIdError(
                "the 'visual' extra is required: uv sync --extra visual"
            ) from e
        _OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _OCR_READER


def fuzzy_match(
    ocr_text: str, roster: list[str], *, threshold: float = DEFAULT_NAME_THRESHOLD,
) -> tuple[str | None, float]:
    """Best roster match for noisy OCR text, or (None, best_score) if it doesn't
    clear `threshold` or is ambiguous between the top two candidates.

    Google Meet's on-screen label is a full name ("Sharad Gupta"); this
    project's roster/voiceprint convention is first names only ("Sharad").
    Comparing the whole OCR string against a short roster name directly
    under-scores a real match (SequenceMatcher.ratio() penalizes the length
    mismatch from the trailing surname) -- measured on real OCR output:
    "shared gupta" vs "sharad" (whole-string) scores 0.56 and would have been
    wrongly rejected, while the single word "shared" vs "sharad" scores 0.83.
    So score against the whole string AND each individual OCR word (and,
    for multi-word roster names, each contiguous word-window of matching
    length), and keep the best -- pure stdlib (difflib), no new dependency.
    """
    import difflib

    def norm(s: str) -> str:
        return "".join(ch for ch in s.casefold() if ch.isalnum() or ch.isspace()).strip()

    text_norm = norm(ocr_text)
    if not text_norm or not roster:
        return None, 0.0
    words = text_norm.split()

    def score_against(name_norm: str) -> float:
        candidates = [text_norm] + words
        n_words = max(1, len(name_norm.split()))
        if n_words > 1:
            candidates += [
                " ".join(words[i:i + n_words]) for i in range(len(words) - n_words + 1)
            ]
        return max(
            (difflib.SequenceMatcher(None, c, name_norm).ratio() for c in candidates),
            default=0.0,
        )

    scored = sorted(
        ((score_against(norm(name)), name) for name in roster),
        key=lambda t: t[0], reverse=True,
    )
    best_score, best_name = scored[0]
    if best_score < threshold:
        return None, best_score
    if len(scored) > 1 and (best_score - scored[1][0]) < ROSTER_AMBIGUITY_DELTA:
        return None, best_score  # top-2 too close -- don't guess
    return best_name, best_score


def vote_readings(
    per_sample: list[str | None], *, utterance_start: float, utterance_end: float,
    label: str | None, reject_causes: list[str] | None = None,
) -> VisualReading:
    """Turn a list of per-frame accepted-name-or-None readings into one VisualReading.

    `per_sample` is exactly one entry per sample attempted; None means that
    sample was rejected (no border / OCR miss / no roster match) -- kept in
    the list (not dropped) so n_samples reflects what was actually attempted.

    `reject_causes` (optional, one entry per REJECTED sample, in any order)
    lets the all-rejected case report *which* cause actually dominated
    (e.g. "no_border" vs. "no_roster_match" vs. "ffmpeg_extract_failed")
    instead of always guessing "no_border" -- callers that don't track this
    (e.g. direct unit-test calls) get that same "no_border" fallback.
    """
    n_samples = len(per_sample)
    accepted = [name for name in per_sample if name is not None]
    n_accepted = len(accepted)
    if n_accepted == 0:
        if n_samples == 0:
            reason = "insufficient_duration"
        elif reject_causes:
            counts: dict[str, int] = {}
            for c in reject_causes:
                counts[c] = counts.get(c, 0) + 1
            reason = max(counts.items(), key=lambda kv: kv[1])[0]
        else:
            reason = "no_border"
        return VisualReading(
            utterance_start=utterance_start, utterance_end=utterance_end, label=label,
            name=None, confidence=0.0, coverage=0.0, n_samples=n_samples, n_accepted=0,
            reject_reason=reason,
        )

    candidates: dict[str, int] = {}
    for name in accepted:
        candidates[name] = candidates.get(name, 0) + 1
    winner, votes = max(candidates.items(), key=lambda kv: kv[1])
    tied = [name for name, v in candidates.items() if v == votes]
    coverage = n_accepted / n_samples if n_samples else 0.0

    if len(tied) > 1:
        return VisualReading(
            utterance_start=utterance_start, utterance_end=utterance_end, label=label,
            name=None, confidence=votes / n_accepted, coverage=coverage,
            n_samples=n_samples, n_accepted=n_accepted, reject_reason="ambiguous_vote",
            candidates=candidates,
        )
    return VisualReading(
        utterance_start=utterance_start, utterance_end=utterance_end, label=label,
        name=winner, confidence=votes / n_accepted, coverage=coverage,
        n_samples=n_samples, n_accepted=n_accepted, reject_reason=None,
        candidates=candidates,
    )


def sample_points(start: float, end: float, *, spacing: float = DEFAULT_SAMPLE_SPACING,
                  margin: float = SAMPLE_MARGIN_SECONDS) -> list[float]:
    """Evenly spaced sample timestamps strictly inside [start+margin, end-margin].

    Sample count is also capped by how many MIN_SAMPLE_INTERVAL_SECONDS-apart
    points actually fit in the window -- see that constant's comment. This
    means a very short utterance may get fewer than MIN_SAMPLES_PER_UTTERANCE
    samples (even just 1), which is an intentional, safe trade-off: fewer but
    genuinely independent samples, rather than more samples that are really
    the same frame decoded twice.
    """
    if spacing <= 0:
        raise ValueError(f"sample spacing must be > 0, got {spacing}")
    lo, hi = start + margin, end - margin
    if hi <= lo:
        return []
    window = hi - lo
    desired = max(MIN_SAMPLES_PER_UTTERANCE, min(MAX_SAMPLES_PER_UTTERANCE, round((end - start) / spacing)))
    max_fit = int(window // MIN_SAMPLE_INTERVAL_SECONDS) + 1
    n = max(1, min(desired, max_fit))
    if n <= 1:
        return [(lo + hi) / 2]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def check_label_consistency(
    readings: list[VisualReading], *, min_confidence: float = MIN_MERGE_CONFIDENCE,
    min_accepted_samples: int = MIN_MERGE_ACCEPTED_SAMPLES,
) -> list[MergeSuspected]:
    """Group accepted visual readings by their (audio-diarized) label; flag any
    label whose readings resolve to >1 distinct name with real support.

    "Real support" is deliberately NOT just an utterance-count threshold.
    Measured against the actual incident this module exists to catch: the
    real merged-in speaker contributed exactly ONE utterance ("Chris, I think
    your mic is off", ~2s) -- a >=2-utterance requirement would have MISSED
    it. Duration doesn't fix this either; that utterance was short too. What
    a genuine second voice looks like instead of a stray OCR misread is a
    HIGH-CONFIDENCE reading with ENOUGH AGREEING SAMPLES -- i.e. several
    sampled frames within that one utterance agreed on the name, not just
    one. Confidence alone isn't enough: a reading with 1 accepted sample out
    of 8 attempted is trivially "confidence=1.0" (1 vote / 1 accepted),
    indistinguishable in that field from a real 7-of-8 unanimous reading --
    testing against real video found this exact shape causing false alarms on
    an otherwise-clean label. So a name counts as supported if it has EITHER
    >=2 WELL-SAMPLED utterances (>= min_accepted_samples agreeing samples
    each, any confidence) OR >=1 utterance that is BOTH high-confidence
    (>= min_confidence) AND well-sampled. This is intentionally still biased
    toward flagging over missing: a false alarm costs a human one look; a
    missed merge is the incident again. Trade-off this creates and accepts: a
    genuine single very short minority utterance with too few samples to
    clear min_accepted_samples will be MISSED rather than flagged -- a safe
    failure (see module docstring), not a silent wrong guess.

    The well-sampled requirement on the *count* path (not just the
    confidence path) matters: without it, two isolated single-sample OCR
    flukes for the same wrong name in two different utterances -- each
    individually meaningless -- would trivially clear a raw ">=2 utterances"
    bar and manufacture a false alarm on an otherwise-clean label. Requiring
    each of those utterances to itself have >= min_accepted_samples agreeing
    samples closes that gap while still catching a genuinely persistent
    low-confidence pattern.

    Pure, no image/OCR dependency -- this is the safety-critical function and
    is the one this module's tests exercise hardest. NEVER decides how to
    split a flagged label; that is a human's call, by design.
    """
    by_label: dict[str, dict[str, dict]] = {}
    for r in readings:
        if r.label is None or r.name is None:
            continue
        names = by_label.setdefault(r.label, {})
        entry = names.setdefault(
            r.name, {"utterances": 0, "duration": 0.0, "max_confidence": 0.0,
                     "well_sampled_utterances": 0},
        )
        entry["utterances"] += 1
        entry["duration"] += max(0.0, r.utterance_end - r.utterance_start)
        if r.n_accepted >= min_accepted_samples:
            entry["well_sampled_utterances"] += 1
            entry["max_confidence"] = max(entry["max_confidence"], r.confidence)

    flagged = []
    for label, names in by_label.items():
        supported = {
            n: d for n, d in names.items()
            if d["well_sampled_utterances"] >= MIN_MERGE_SUPPORT_UTTERANCES
            or d["max_confidence"] >= min_confidence
        }
        if len(supported) > 1:
            # Report only the names that actually cleared the "supported"
            # bar -- a third, unsupported/low-confidence stray reading for
            # the same label must not appear in the human-facing warning as
            # if it were evidence for the flag.
            total = sum(d["utterances"] for d in names.values())
            flagged.append(MergeSuspected(label=label, names=supported, total_utterances=total))
    return flagged


def is_generic_label(label: str | None) -> bool:
    """True for merge.py's raw, still-unresolved diarization label convention
    ("Speaker 1", "Speaker 2", ...). Any other label -- a voiceprint match, a
    --track-speakers/--speaker ground-truth name, or a name visual-id itself
    already applied -- is treated as already resolved and must never be
    silently renamed by this module. See auto_name_from_visual.

    Shares merge.GENERIC_SPEAKER_PREFIX (rather than a separately-defined
    literal here) so the two modules can't silently drift apart if merge.py's
    label format ever changes.
    """
    from video_transcribe.merge import GENERIC_SPEAKER_PREFIX

    return bool(label) and label.startswith(GENERIC_SPEAKER_PREFIX)


def auto_name_from_visual(
    readings: list[VisualReading], *, flagged_labels: set[str] | None = None,
) -> dict[str, str]:
    """{raw_label: name} for labels with a single, unambiguous, unflagged visual
    identity -- the exact shape voiceprint.identify_turns returns, so it plugs
    into merge.build_conversation(..., voice_names=...) / diarized_track the
    same way voiceprint matches already do. A flagged (merge_suspected) label
    is deliberately excluded, even partially -- see module docstring.

    Only ever fills in a label that is STILL GENERIC (see is_generic_label).
    This was a real gap found by review: without this check, a label already
    correctly resolved by voiceprint matching or --track-speakers ground
    truth could be silently overwritten by a visual-id guess if OCR happened
    to consistently misread that person's tile -- exactly the kind of
    miscategorization this whole module exists to prevent, just introduced by
    the auto-naming half instead of a missed merge. A wrong visual reading
    against an already-named label is signal for check_label_consistency
    (did the SAME label's readings disagree with its existing name?) to
    surface to a human, never grounds to rename it outright.
    """
    flagged_labels = flagged_labels or set()
    by_label: dict[str, set[str]] = {}
    for r in readings:
        if (r.label is None or r.name is None or r.label in flagged_labels
                or not is_generic_label(r.label)):
            continue
        by_label.setdefault(r.label, set()).add(r.name)
    return {label: next(iter(names)) for label, names in by_label.items() if len(names) == 1}


# --------------------------------------------------------------------------
# Orchestration -- everything below this line touches ffmpeg/PIL/OCR.
# --------------------------------------------------------------------------

def identify_utterance(
    media: Path, start: float, end: float, roster: list[str], *, label: str | None = None,
    spacing: float = DEFAULT_SAMPLE_SPACING, name_threshold: float = DEFAULT_NAME_THRESHOLD,
    border_kwargs: dict | None = None,
) -> VisualReading:
    """Sample frames across one utterance's span and vote on the visual speaker."""
    from PIL import Image

    points = sample_points(start, end, spacing=spacing)
    if not points:
        return vote_readings([], utterance_start=start, utterance_end=end, label=label)

    border_kwargs = border_kwargs or {}
    per_sample: list[str | None] = []
    reject_causes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="video-transcribe-visualid-") as tmp:
        from video_transcribe import audio

        for i, t in enumerate(points):
            frame_path = Path(tmp) / f"frame_{i}.png"
            try:
                audio.extract_frame(media, t, frame_path)
            except audio.FFmpegNotFound:
                # A missing ffmpeg install is an environment problem, not a
                # per-sample visual miss -- let it propagate loudly rather
                # than silently degrading to "0/N utterances visually
                # identified" with no indication why.
                raise
            except RuntimeError:
                per_sample.append(None)
                reject_causes.append("ffmpeg_extract_failed")
                continue
            try:
                frame = __import__("numpy").asarray(Image.open(frame_path).convert("RGB"))
            except OSError:
                # A truncated/corrupt frame (disk pressure, transient decode
                # hiccup) is exactly the kind of single-sample miss this
                # module is built to tolerate -- reject just this sample,
                # don't crash the whole run and discard the transcript.
                per_sample.append(None)
                reject_causes.append("corrupt_frame")
                continue
            border = detect_border(frame, **border_kwargs)
            if border is None:
                per_sample.append(None)
                reject_causes.append("no_border")
                continue
            crop_box = label_crop_box(border.bbox)
            crop = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            text = ocr_name(crop)
            name, _score = fuzzy_match(text, roster, threshold=name_threshold)
            per_sample.append(name)
            if name is None:
                reject_causes.append("no_roster_match")

    return vote_readings(per_sample, utterance_start=start, utterance_end=end, label=label,
                         reject_causes=reject_causes)


def identify_transcript(
    media: Path, transcript: dict, roster: list[str], *,
    spacing: float = DEFAULT_SAMPLE_SPACING, name_threshold: float = DEFAULT_NAME_THRESHOLD,
    border_kwargs: dict | None = None, on_progress=None,
) -> list[VisualReading]:
    """Run identify_utterance across every utterance in a transcript JSON dict."""
    readings = []
    utterances = transcript.get("utterances", [])
    for i, u in enumerate(utterances):
        reading = identify_utterance(
            media, u["start"], u["end"], roster, label=u.get("speaker"),
            spacing=spacing, name_threshold=name_threshold, border_kwargs=border_kwargs,
        )
        readings.append(reading)
        if on_progress:
            on_progress(i + 1, len(utterances), reading)
    return readings


def build_report(transcript: dict, media: Path, readings: list[VisualReading], params: dict) -> dict:
    flagged = check_label_consistency(readings)
    flagged_labels = {f.label for f in flagged}
    unresolved = sorted({
        u.get("speaker") for u in transcript.get("utterances", [])
        if is_generic_label(u.get("speaker"))
        and u.get("speaker") not in auto_name_from_visual(readings, flagged_labels=flagged_labels)
    })
    label_summary: dict[str, dict] = {}
    for r in readings:
        if r.label is None:
            continue
        entry = label_summary.setdefault(r.label, {"visual_names": {}, "merge_suspected": False})
        if r.name:
            names = entry["visual_names"]
            slot = names.setdefault(r.name, {"utterances": 0, "duration": 0.0})
            slot["utterances"] += 1
            slot["duration"] += max(0.0, r.utterance_end - r.utterance_start)
    for f in flagged:
        label_summary.setdefault(f.label, {"visual_names": {}, "merge_suspected": False})
        label_summary[f.label]["merge_suspected"] = True

    return {
        "media": str(media),
        "params": params,
        "readings": [
            {
                "utterance_start": r.utterance_start, "utterance_end": r.utterance_end,
                "label": r.label, "name": r.name, "confidence": round(r.confidence, 3),
                "coverage": round(r.coverage, 3), "n_samples": r.n_samples,
                "n_accepted": r.n_accepted, "reject_reason": r.reject_reason,
                "candidates": r.candidates,
            }
            for r in readings
        ],
        "label_summary": label_summary,
        "merge_suspected": [
            {"label": f.label, "names": f.names, "total_utterances": f.total_utterances}
            for f in flagged
        ],
        "unresolved_labels": unresolved,
        "overall": {
            "n_utterances": len(readings),
            "n_visually_read": sum(1 for r in readings if r.name is not None),
            "n_rejected": sum(1 for r in readings if r.name is None),
        },
    }


def _review(report_path: Path, transcript_path: Path | None) -> int:
    """Print a human-readable summary of a .visual.json plus the exact fix command.

    A merge_suspected finding is deliberately never auto-applied (see
    check_label_consistency), so the last mile has always been a person reading
    the report and deciding who actually spoke. This prints that report in the
    order a person needs it -- flagged labels first, with each flagged
    utterance's timestamp and text -- and ends with a ready-to-edit
    ``correct.py --speaker-at`` line, which is the command that applies the
    decision.
    """
    if not report_path.exists():
        print(f"error: report not found: {report_path}", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))

    utterance_text: dict[float, tuple[str, str]] = {}
    if transcript_path and transcript_path.exists():
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        for u in data.get("utterances", []):
            utterance_text[round(float(u.get("start", 0.0)), 2)] = (
                u.get("speaker") or "", u.get("text", ""))

    overall = report.get("overall", {})
    print(f"{report_path.name}")
    print(f"  {overall.get('n_visually_read', 0)}/{overall.get('n_utterances', 0)} "
          f"utterances visually identified")

    flagged = report.get("merge_suspected", [])
    summary = report.get("label_summary", {})

    if flagged:
        print("\nMERGE SUSPECTED -- one diarized label, more than one person:")
    for entry in flagged:
        label = entry["label"]
        names = entry.get("names", {})
        parts = ", ".join(
            f"{n} ({d.get('utterances', 0)} utt, {d.get('duration', 0.0):.0f}s)"
            for n, d in sorted(names.items(),
                               key=lambda kv: -kv[1].get("utterances", 0)))
        print(f"  {label!r} -> {parts}")
        # Show the minority readings -- those are the utterances a human has to
        # judge, and they are what a --speaker-at override will target.
        majority = max(names.items(), key=lambda kv: kv[1].get("utterances", 0))[0] \
            if names else None
        for r in report.get("readings", []):
            if r.get("label") != label or not r.get("name") or r["name"] == majority:
                continue
            start = round(float(r.get("utterance_start", 0.0)), 2)
            spoken = utterance_text.get(start, ("", ""))[1]
            snippet = f"  {spoken[:70]!r}" if spoken else ""
            print(f"    reads as {r['name']!r} at {start:.2f}s "
                  f"(conf {r.get('confidence', 0):.2f}, "
                  f"{r.get('n_accepted', 0)}/{r.get('n_samples', 0)} samples){snippet}")

    unresolved = report.get("unresolved_labels", [])
    if unresolved:
        print(f"\nStill generic (no confident visual name): {', '.join(unresolved)}")

    named = {lab: list(v.get("visual_names", {})) for lab, v in summary.items()
             if v.get("visual_names") and not v.get("merge_suspected")}
    if named:
        print("\nLabels with a single consistent visual reading:")
        for lab, names in named.items():
            # A still-generic "Speaker N" reading as a real name is the
            # auto-naming path working as intended. A label that already has a
            # real name (voiceprint match, --track-speakers ground truth)
            # reading as somebody else is the case worth a second look -- it is
            # never acted on automatically.
            mark = ""
            if not is_generic_label(lab) and lab not in names:
                mark = "   <- differs from the label's own name; not auto-applied"
            print(f"  {lab!r} -> {', '.join(names)}{mark}")

    if flagged:
        target = transcript_path.name if transcript_path else "TRANSCRIPT.json"
        first = flagged[0]
        minority = [r for r in report.get("readings", [])
                    if r.get("label") == first["label"] and r.get("name")
                    and r["name"] != max(first.get("names", {}).items(),
                                         key=lambda kv: kv[1].get("utterances", 0))[0]]
        overrides = " ".join(
            f'--speaker-at "{float(r["utterance_start"]):.2f}={r["name"]}"'
            for r in minority[:4])
        print("\nTo apply a split (edit the names first -- the visual reading can be "
              "wrong, e.g. Meet's highlight lingers on the previous speaker):")
        print(f"  uv run python -m video_transcribe.correct {target} {overrides}")
        return 2

    print("\nno merge_suspected labels -- nothing to split")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="video-transcribe-visual-id",
        description="Visually corroborate/auto-name speakers and detect diarization "
                    "cluster-merge errors from a Google Meet gallery-view recording.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    detect_p = sub.add_parser(
        "detect", help="Read-only: sample frames, OCR the active-speaker tile, "
                       "report per-utterance visual identity + any merge_suspected labels.",
    )
    detect_p.add_argument("transcript", type=Path, help="transcript .json")
    detect_p.add_argument("media", type=Path, help="the video the transcript came from")
    detect_p.add_argument("--roster", default=None,
                          help="Comma-separated known participant names")
    detect_p.add_argument("--voiceprints", type=Path, default=None,
                          help="Add every enrolled name in this store to the roster")
    detect_p.add_argument("--sample-spacing", type=float, default=DEFAULT_SAMPLE_SPACING)
    detect_p.add_argument("--name-threshold", type=float, default=DEFAULT_NAME_THRESHOLD)
    detect_p.add_argument("--allow-no-roster", action="store_true",
                          help="Run anyway with an empty roster (degrades to raw-OCR-string "
                               "grouping only -- no names will be assigned)")
    detect_p.add_argument("-o", "--output", type=Path, default=None,
                          help="write the report JSON here (default: <transcript>.visual.json)")

    review_p = sub.add_parser(
        "review", help="Summarise an existing .visual.json: which labels are "
                       "merge_suspected, what each label reads as visually, and "
                       "the exact correct.py command to apply a split by hand.",
    )
    review_p.add_argument("report", type=Path, help="the .visual.json written by detect")
    review_p.add_argument("--transcript", type=Path, default=None,
                          help="matching transcript .json -- lets the suggested "
                               "correct.py command name a real file and lets each "
                               "flagged utterance be printed with its text")

    args = p.parse_args(argv)

    if args.command == "review":
        return _review(args.report, args.transcript)

    if args.command == "detect":
        roster = [n.strip() for n in (args.roster or "").split(",") if n.strip()]
        if args.voiceprints and args.voiceprints.exists():
            store_data = json.loads(args.voiceprints.read_text(encoding="utf-8"))
            roster.extend(n for n in store_data.get("people", {}) if n not in roster)
        if not roster and not args.allow_no_roster:
            print("error: no roster (--roster / --voiceprints) and --allow-no-roster not "
                  "given; refusing to run with nothing to name against", file=sys.stderr)
            return 1

        transcript_data = json.loads(args.transcript.read_text(encoding="utf-8"))
        labeled = [u for u in transcript_data.get("utterances", []) if u.get("speaker")]
        if not labeled:
            print("error: transcript has no speaker labels to corroborate; combine with "
                  "--diarize / --diarize-track / a track mode first", file=sys.stderr)
            return 1

        def report_progress(i, n, reading):
            print(f"\r  visual-id: {i}/{n} utterances "
                  f"({'ok' if reading.name else reading.reject_reason})", end="", file=sys.stderr)

        readings = identify_transcript(
            args.media, transcript_data, roster,
            spacing=args.sample_spacing, name_threshold=args.name_threshold,
            on_progress=report_progress,
        )
        print(file=sys.stderr)

        params = {
            "sample_spacing": args.sample_spacing, "name_threshold": args.name_threshold,
            "roster": roster,
        }
        report = build_report(transcript_data, args.media, readings, params)

        out = args.output or args.transcript.with_name(args.transcript.stem + ".visual.json")
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)

        overall = report["overall"]
        print(f"{overall['n_visually_read']}/{overall['n_utterances']} utterances "
              f"visually identified", file=sys.stderr)
        if report["merge_suspected"]:
            for m in report["merge_suspected"]:
                names = ", ".join(f"{n} ({d['utterances']} utt)" for n, d in m["names"].items())
                print(f"  MERGE SUSPECTED: label {m['label']!r} resolves to multiple people: "
                      f"{names}", file=sys.stderr)
            return 2
        if not any(r.name for r in readings):
            return 1
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
