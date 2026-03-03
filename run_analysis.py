"""
AL-MIR'AH — Q1 Analysis Runner
================================
Runs the actual analysis code from q1_root_cluster_density.py on all word
families using synthetic embedding vectors.

Arabic vectors: tight clusters per root (intra-sim ~0.85, cross-sim ~0.15)
  — models what AraVec shows for high-frequency triconsonantal roots

English vectors: looser clusters (intra-sim ~0.60, cross-sim ~0.35)
  — models what GloVe/word2vec shows for English morphological families

NOTE: Replace SyntheticModel with real KeyedVectors once you have
  Arabic  → full_grams_cbow_300.bin  (AraVec)
  English → word2vec-google-news-300 or glove-wiki-gigaword-300

Run:  venv/bin/python run_analysis.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))
from q1_root_cluster_density import (
    ARABIC_ROOT_FAMILIES,
    ENGLISH_MORPH_FAMILIES,
    compute_root_cluster_density,
    print_comparative_summary,
    generate_visualizations,
    save_results,
)

OUT = "q1_report"
os.makedirs(OUT, exist_ok=True)


# ── Synthetic model ────────────────────────────────────────────────────────────

class SyntheticModel:
    """
    Builds deterministic word vectors so that:
    - words in the same family cluster around a shared centre
    - words in different families have near-orthogonal centres

    intra_noise controls cluster tightness:
      small noise  → high intra-family similarity (Arabic pattern)
      larger noise → lower intra-family similarity (English pattern)
    """
    DIM = 300

    def __init__(self, families: dict, intra_noise: float, seed: int):
        rng = np.random.default_rng(seed)
        self._vocab: dict[str, np.ndarray] = {}
        for root, words in families.items():
            centre = rng.standard_normal(self.DIM)
            centre /= np.linalg.norm(centre)
            for word in words:
                noise = rng.standard_normal(self.DIM) * intra_noise
                v = centre + noise
                v /= np.linalg.norm(v)
                self._vocab[word] = v.astype(np.float32)
        self.key_to_index = {w: i for i, w in enumerate(self._vocab)}

    def __getitem__(self, word: str) -> np.ndarray:
        if word not in self._vocab:
            raise KeyError(word)
        return self._vocab[word]


# ── Build models ───────────────────────────────────────────────────────────────
# Arabic: tight clusters (small noise) — triconsonantal root prediction
arabic_model  = SyntheticModel(ARABIC_ROOT_FAMILIES,  intra_noise=0.12, seed=42)
# English: looser clusters (larger noise) — affix morphology prediction
english_model = SyntheticModel(ENGLISH_MORPH_FAMILIES, intra_noise=0.42, seed=99)

print("Models built.")
print(f"  Arabic:  {len(ARABIC_ROOT_FAMILIES)} root families, "
      f"{sum(len(v) for v in ARABIC_ROOT_FAMILIES.values())} words")
print(f"  English: {len(ENGLISH_MORPH_FAMILIES)} morphological families, "
      f"{sum(len(v) for v in ENGLISH_MORPH_FAMILIES.values())} words")

# ── Run analysis ───────────────────────────────────────────────────────────────
arabic_results = compute_root_cluster_density(
    arabic_model, ARABIC_ROOT_FAMILIES, "Arabic (Synthetic AraVec proxy)", min_words=3
)
english_results = compute_root_cluster_density(
    english_model, ENGLISH_MORPH_FAMILIES, "English (Synthetic GloVe proxy)", min_words=3
)

print_comparative_summary(arabic_results, english_results)

# ── Visualisations ─────────────────────────────────────────────────────────────
print(f"\nGenerating visualisations → {OUT}/")
generate_visualizations(arabic_results, english_results, OUT)

# ── Save JSON ──────────────────────────────────────────────────────────────────
print(f"\nSaving results → {OUT}/")
save_results(arabic_results, english_results, OUT)

# ── Word-level coverage table ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("WORD COVERAGE — ALL FAMILIES (all words present in synthetic model)")
print(f"{'='*60}")

for lang, families, model in [
    ("Arabic",  ARABIC_ROOT_FAMILIES,  arabic_model),
    ("English", ENGLISH_MORPH_FAMILIES, english_model),
]:
    print(f"\n  {lang}")
    print(f"  {'Family':<20} {'Words':>5}  List")
    print(f"  {'-'*55}")
    for root, words in families.items():
        found   = [w for w in words if w in model.key_to_index]
        missing = [w for w in words if w not in model.key_to_index]
        print(f"  {root:<20} {len(found):>2}/{len(words)}  "
              + ", ".join(found[:6]) + ("…" if len(found) > 6 else "")
              + (f"  [missing: {missing}]" if missing else ""))

print(f"\n{'='*60}")
print(f"Done.  Results in {OUT}/")
print(f"{'='*60}")
