"""One-off evaluation harness for voiceprint matching variants.

Computes embeddings for every utterance of a ground-truth transcript once,
caches them to .npz, then scores matching variants (raw cosine vs centroid,
length-normalized centroids, s-norm) against the confirmed labels so scoring
changes can be measured in seconds without re-extracting audio.

Usage:
  uv run --no-sync python tests/eval_matching.py TRANSCRIPT.json MEDIA \
      --store voiceprints.json [--cache eval_cache.npz]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_transcribe import voiceprint as vp  # noqa: E402


def compute_or_load(transcript: Path, media: Path, cache: Path):
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return data["embeddings"], list(data["labels"]), list(data["starts"])
    embedder = vp.load_embedder()
    embs, labels, starts = [], [], []
    for u, vec in vp._iter_utterance_embeddings(transcript, media, embedder):
        if not u.get("speaker"):
            continue
        embs.append(vec)
        labels.append(u["speaker"])
        starts.append(u["start"])
    embeddings = np.array(embs)
    np.savez(cache, embeddings=embeddings, labels=np.array(labels), starts=np.array(starts))
    return embeddings, labels, starts


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def evaluate(name: str, scores_by_person: dict[str, np.ndarray], labels: list[str],
             thresholds: list[float]) -> None:
    """scores_by_person: {person: score-per-utterance array}."""
    people = sorted(scores_by_person)
    mat = np.stack([scores_by_person[p] for p in people])  # (people, utts)
    pred_idx = mat.argmax(axis=0)
    pred_score = mat.max(axis=0)
    best = None
    for th in thresholds:
        correct = wrong = miss = 0
        for i, true in enumerate(labels):
            if true not in scores_by_person:
                continue
            if pred_score[i] < th:
                miss += 1
            elif people[pred_idx[i]] == true:
                correct += 1
            else:
                wrong += 1
        total = correct + wrong + miss
        acc = correct / total if total else 0.0
        if best is None or acc > best[1]:
            best = (th, acc, correct, wrong, miss, total)
    th, acc, correct, wrong, miss, total = best
    print(f"{name:<42} best_th={th:.2f}  acc={acc * 100:5.1f}%  "
          f"correct={correct} wrong={wrong} miss={miss} / {total}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("transcript", type=Path)
    p.add_argument("media", type=Path)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path(__file__).parent / "eval_cache.npz")
    args = p.parse_args()

    embeddings, labels, _starts = compute_or_load(args.transcript, args.media, args.cache)
    store = vp.VoiceprintStore.load(args.store)
    people = sorted(store.people)
    print(f"{len(labels)} labeled utterances; store: "
          + ", ".join(f"{n}({len(store.people[n])})" for n in people))

    test = np.stack([unit(e) for e in embeddings])  # (utts, dim)

    # --- variant 1: raw cosine vs plain centroid (current behavior) ---
    cents_raw = {n: unit(np.mean(np.array(store.people[n]), axis=0)) for n in people}
    v1 = {n: test @ cents_raw[n] for n in people}
    evaluate("raw cosine vs centroid (baseline)", v1, labels,
             [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])

    # --- variant 2: cosine vs length-normalized-before-averaging centroid ---
    cents_ln = {n: unit(np.mean(np.stack([unit(s) for s in store.people[n]]), axis=0))
                for n in people}
    v2 = {n: test @ cents_ln[n] for n in people}
    evaluate("cosine vs length-normed centroid", v2, labels,
             [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])

    # --- variant 3: nearest individual sample (max over samples) ---
    v3 = {n: np.max(np.stack([unit(s) for s in store.people[n]]) @ test.T, axis=0)
          for n in people}
    evaluate("nearest-sample cosine", v3, labels,
             [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])

    # --- variant 4: s-norm over length-normed centroids ---
    # enroll-side cohort for person n: all *other* people's samples;
    # test-side cohort: all enrolled samples.
    all_samples = {n: np.stack([unit(s) for s in store.people[n]]) for n in people}
    v4 = {}
    for n in people:
        cohort_e = np.concatenate([all_samples[m] for m in people if m != n])
        e_scores = cohort_e @ cents_ln[n]                    # centroid vs impostors
        mu_e, sd_e = e_scores.mean(), max(e_scores.std(), 1e-6)
        raw = test @ cents_ln[n]
        cohort_t = np.concatenate([all_samples[m] for m in people])
        t_scores = test @ cohort_t.T                          # (utts, cohort)
        mu_t, sd_t = t_scores.mean(axis=1), np.maximum(t_scores.std(axis=1), 1e-6)
        v4[n] = 0.5 * ((raw - mu_e) / sd_e + (raw - mu_t) / sd_t)
    evaluate("s-norm (length-normed centroids)", v4, labels,
             [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

    # --- variant 5: s-norm + leave-own-samples-out test cohort ---
    v5 = {}
    for n in people:
        cohort_e = np.concatenate([all_samples[m] for m in people if m != n])
        e_scores = cohort_e @ cents_ln[n]
        mu_e, sd_e = e_scores.mean(), max(e_scores.std(), 1e-6)
        raw = test @ cents_ln[n]
        cohort_t = np.stack([cents_ln[m] for m in people if m != n])
        t_scores = test @ cohort_t.T
        mu_t, sd_t = t_scores.mean(axis=1), np.maximum(t_scores.std(axis=1), 1e-6)
        v5[n] = 0.5 * ((raw - mu_e) / sd_e + (raw - mu_t) / sd_t)
    evaluate("s-norm (other-centroids test cohort)", v5, labels,
             [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

    # ------------------------------------------------------------------
    # Cluster-level: the REAL deployment path. identify_turns averages a
    # whole diarized cluster before matching; simulate with perfect
    # clusters (group utterances by true speaker), then match each cluster
    # both independently and with exclusive one-to-one greedy assignment.
    # ------------------------------------------------------------------
    print("\n--- cluster level (group-by-true-speaker, mean embedding) ---")
    cluster_names = sorted(set(labels))
    clusters = {c: unit(np.mean(test[[i for i, l in enumerate(labels) if l == c]], axis=0))
                for c in cluster_names}

    def cluster_scores(cent_map):
        return {c: {n: float(clusters[c] @ cent_map[n]) for n in people}
                for c in cluster_names}

    def report_clusters(name, sc, threshold):
        # independent argmax
        indep = {c: max(sc[c], key=sc[c].get) if max(sc[c].values()) >= threshold else None
                 for c in cluster_names}
        # exclusive greedy: highest score first, each person used once
        pairs = sorted(((sc[c][n], c, n) for c in cluster_names for n in people),
                       reverse=True)
        excl, used_c, used_n = {}, set(), set()
        for s, c, n in pairs:
            if s < threshold or c in used_c or n in used_n:
                continue
            excl[c] = n
            used_c.add(c)
            used_n.add(n)
        ind_ok = sum(1 for c in cluster_names if indep.get(c) == c)
        exc_ok = sum(1 for c in cluster_names if excl.get(c) == c)
        print(f"{name:<42} independent={ind_ok}/{len(cluster_names)}  "
              f"exclusive={exc_ok}/{len(cluster_names)}")
        detail = {c: (indep.get(c), excl.get(c)) for c in cluster_names}
        print(f"    (true -> indep / excl): "
              + "; ".join(f"{c}->{i}/{e}" for c, (i, e) in detail.items()))

    report_clusters("cluster cosine (raw centroids, th=0.5)", cluster_scores(cents_raw), 0.5)
    report_clusters("cluster cosine (ln centroids, th=0.5)", cluster_scores(cents_ln), 0.5)
    report_clusters("cluster cosine (ln centroids, th=0.0)", cluster_scores(cents_ln), 0.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
