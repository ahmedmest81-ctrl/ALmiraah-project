"""
q2_robustness_freq_matched.py
─────────────────────────────────────────────────────────────────────────
Frequency-matched permutation null for the Arabic Mashriqi Abjad-attention
result. Shuffles letter values WITHIN frequency-matched groups, not across
the full 28-letter set. This isolates the Mashriqi-structural component
from the letter-frequency-baseline component.

Background:
─────────────
The standard shuffled-Abjad permutation null (in q2_robustness.py) gave:
  observed |ρ| = 0.0558
  permuted mean = −0.030 (NOT zero; SD = 0.013)
  empirical p = 0.026

The non-zero permuted mean indicates that random letter-value mappings
produce a baseline correlation in attention geometry of about ρ = −0.030,
arising from letter-frequency structure interacting with attention through
nonlinear pathways the partial-correlation controls don't eliminate.

The Mashriqi-specific component above this baseline is approximately
ρ = 0.026. We want to test whether this 0.026 increment is preserved
when we restrict the permutation to letter-frequency-matched groups —
i.e. when we ask "does Mashriqi outperform random Mashriqi-like mappings
that preserve letter-frequency structure?"

The frequency-matched null:
  • Compute corpus frequency for each of the 28 base Arabic letters
  • Sort letters by frequency, partition into K equal-size bins (K=4 or K=7)
  • Within each bin, shuffle the Mashriqi values among the letters in that bin
  • This preserves: which letters are most frequent, how Mashriqi values are
    distributed across frequency tiers
  • This breaks: the specific Mashriqi-letter-to-value assignment within tiers

If the Mashriqi-specific component is real and structural, the
frequency-matched permutation null should still produce ρ values smaller
than the observed −0.0558 with high probability — because the actual
Mashriqi assignment is doing something specific within frequency tiers.

If the frequency-matched null produces ρ values comparable to or larger
than observed, the apparent Mashriqi specificity is an artifact of how
Mashriqi values happen to align with letter-frequency tiers, not of any
within-tier letter-to-value structure.

Pre-registered decision rule:
  • PASS (Mashriqi-structural): empirical p (two-tailed) < 0.05 against
    K=4 frequency-matched null. Headline finding strengthened: Mashriqi
    is not just frequency-tier-structured but has within-tier specificity.
  • FAIL: empirical p ≥ 0.05. The Mashriqi-specific component is
    primarily about frequency-tier alignment, not within-tier structure.
  • Sensitivity: report results for K=4 and K=7 separately.

Usage:
  python q2_robustness_freq_matched.py \\
      --records q2_results/q2_records.pkl \\
      --cooc q2_results/q2_cooc.pkl \\
      --existing-results q2_results/q2_results_with_robustness.json \\
      --output q2_results/q2_freq_matched_null.json \\
      --n-permutations 1000 \\
      --bins 4

Runtime estimate: same order as q2_robustness.py, approximately
8-10 hours per K-value on consumer CPU. Run K=4 first; K=7 only if K=4
produces an interesting result (PASS or near-PASS).
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Windows console UTF-8 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Mashriqi values — canonical assignments for the 28 base letters.
# ─────────────────────────────────────────────────────────────────────────
MASHRIQI_VALUES = {
    "ا": 1,    "ب": 2,    "ج": 3,    "د": 4,    "ه": 5,
    "و": 6,    "ز": 7,    "ح": 8,    "ط": 9,    "ي": 10,
    "ك": 20,   "ل": 30,   "م": 40,   "ن": 50,   "س": 60,
    "ع": 70,   "ف": 80,   "ص": 90,   "ق": 100,  "ر": 200,
    "ش": 300,  "ت": 400,  "ث": 500,  "خ": 600,  "ذ": 700,
    "ض": 800,  "ظ": 900,  "غ": 1000,
}

# Orthographic equivalence classes — variant glyphs that map to the same
# base letter (preserves Arabic spelling invariants under permutation).
ORTHOGRAPHIC_EQUIVALENTS = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ة": "ت",
    "ى": "ي",
    "ؤ": "و",
    "ئ": "ي",
    "ـ": "",
}


def normalize_letter(c: str) -> str:
    """Map variant glyphs to their base letter."""
    return ORTHOGRAPHIC_EQUIVALENTS.get(c, c)


def compute_word_abjad(word: str, letter_values: dict) -> int:
    """Compute Abjad value of a word given a letter-to-value mapping."""
    total = 0
    for c in word:
        base = normalize_letter(c)
        if base in letter_values:
            total += letter_values[base]
    return total


# ─────────────────────────────────────────────────────────────────────────
# Letter-frequency computation from the corpus records
# ─────────────────────────────────────────────────────────────────────────
def compute_letter_frequencies(records: list) -> Counter:
    """Count occurrences of each base letter across all words in records."""
    letter_counts = Counter()
    seen_words = set()
    for r in records:
        for word in (r.get("word_i", ""), r.get("word_j", "")):
            if word in seen_words:
                continue
            seen_words.add(word)
            for c in word:
                base = normalize_letter(c)
                if base in MASHRIQI_VALUES:
                    letter_counts[base] += 1
    return letter_counts


def partition_letters_by_frequency(letter_freqs: Counter, n_bins: int) -> list[list[str]]:
    """
    Sort letters by frequency, partition into n_bins equal-sized groups.
    Returns a list of bins, each bin being a list of letter strings.
    With 28 letters and n_bins=4: bins of 7 each.
    With 28 letters and n_bins=7: bins of 4 each.
    """
    letters_sorted = [letter for letter, _ in letter_freqs.most_common()]
    # Defensive: include any letters with zero count at the bottom of the order
    for letter in MASHRIQI_VALUES:
        if letter not in letters_sorted:
            letters_sorted.append(letter)

    bins = []
    bin_size = len(letters_sorted) // n_bins
    remainder = len(letters_sorted) % n_bins
    pos = 0
    for i in range(n_bins):
        size = bin_size + (1 if i < remainder else 0)
        bins.append(letters_sorted[pos : pos + size])
        pos += size
    return bins


def make_freq_matched_permuted_values(bins: list[list[str]], rng: random.Random) -> dict:
    """
    Generate a permuted letter-value mapping by shuffling values WITHIN
    each frequency-matched bin. Letters move within their bin only.
    """
    permuted = {}
    for letter_bin in bins:
        # Get the original Mashriqi values of these letters
        bin_values = [MASHRIQI_VALUES[letter] for letter in letter_bin]
        # Shuffle the values among the letters
        shuffled_values = bin_values.copy()
        rng.shuffle(shuffled_values)
        for letter, val in zip(letter_bin, shuffled_values):
            permuted[letter] = val
    return permuted


# ─────────────────────────────────────────────────────────────────────────
# Partial-Spearman computation (matches q2_robustness.py)
# ─────────────────────────────────────────────────────────────────────────
def partial_spearman(
    y: np.ndarray, x: np.ndarray, controls: np.ndarray
) -> tuple[float, float]:
    """
    Partial Spearman correlation between y and x, controlling for `controls`
    (a 2-D array of shape (n, k)).

    This intentionally matches q2_robustness.py: regress raw y and raw x on
    raw controls, then compute Spearman correlation on the residuals. Do not
    rank-transform before residualization, or the observed rho no longer
    reproduces the locked Q2 result.
    """
    from sklearn.linear_model import LinearRegression

    reg_y = LinearRegression().fit(controls, y)
    res_y = y - reg_y.predict(controls)
    reg_x = LinearRegression().fit(controls, x)
    res_x = x - reg_x.predict(controls)

    rho, p = spearmanr(res_x, res_y)
    return rho, p


def compute_partial_rho(
    abjad_distances: np.ndarray,
    attentions: np.ndarray,
    pos_distances: np.ndarray,
    cooc_freqs: np.ndarray,
) -> float:
    """Wrapper for the standard partial-correlation specification."""
    controls = np.column_stack([pos_distances, cooc_freqs])
    rho, _ = partial_spearman(attentions, abjad_distances, controls)
    return rho


# ─────────────────────────────────────────────────────────────────────────
# Main permutation routine
# ─────────────────────────────────────────────────────────────────────────
def run_freq_matched_null(
    records: list,
    cooc_freq: dict,
    n_permutations: int,
    n_bins: int,
    seed: int = 42,
    checkpoint_path: Path | None = None,
) -> dict:
    """
    Run the frequency-matched permutation null.

    Returns a dict with observed_rho, permuted_rhos, summary statistics,
    and pass/fail interpretation against the pre-registered threshold.
    """
    log.info(f"Computing letter frequencies from {len(records):,} records")
    letter_freqs = compute_letter_frequencies(records)
    log.info(f"  Found {len(letter_freqs)} letters with non-zero frequency")
    for letter, count in letter_freqs.most_common(5):
        log.info(f"    {letter}: {count:,}")

    log.info(f"Partitioning into {n_bins} frequency-matched bins")
    bins = partition_letters_by_frequency(letter_freqs, n_bins)
    for i, b in enumerate(bins):
        sample_freqs = [letter_freqs.get(l, 0) for l in b]
        log.info(
            f"  Bin {i+1} ({len(b)} letters): "
            f"freq range {min(sample_freqs):,}-{max(sample_freqs):,} | "
            f"letters {' '.join(b)}"
        )

    # Filter to cross-root pairs only (matching the patched Q2 artifacts).
    # Historical variants of this pipeline used a boolean `same_root`; the
    # current saved records use `root_relation` with values same/different/unknown.
    cross_root = [r for r in records if r.get("root_relation") == "different"]
    log.info(f"Cross-root pairs: {len(cross_root):,}")

    # Pre-extract arrays for fast permutation
    word_i = [r["word_i"] for r in cross_root]
    word_j = [r["word_j"] for r in cross_root]
    attentions = np.array([r["mean_attention"] for r in cross_root])
    pos_distances = np.array([r["pos_distance"] for r in cross_root])
    cooc_array = np.array(
        [
            cooc_freq.get(
                (r["word_i"], r["word_j"]),
                cooc_freq.get((r["word_j"], r["word_i"]), 0.0),
            )
            for r in cross_root
        ]
    )

    # Compute observed ρ with the actual Mashriqi values
    log.info("Computing observed ρ with canonical Mashriqi mapping")
    abjad_i = np.array([compute_word_abjad(w, MASHRIQI_VALUES) for w in word_i])
    abjad_j = np.array([compute_word_abjad(w, MASHRIQI_VALUES) for w in word_j])
    abjad_dist = np.abs(abjad_i - abjad_j)
    observed_rho = compute_partial_rho(abjad_dist, attentions, pos_distances, cooc_array)
    log.info(f"  Observed ρ = {observed_rho:.5f}")

    # Resume from checkpoint if available
    permuted_rhos = []
    start_idx = 0
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            ckpt = json.load(f)
        if ckpt.get("n_bins") == n_bins and ckpt.get("seed") == seed:
            permuted_rhos = ckpt["permuted_rhos"]
            start_idx = ckpt["done"]
            log.info(f"Resuming from checkpoint: {start_idx}/{n_permutations} done")

    # Pre-cache word abjad values per permutation for speed
    rng = random.Random(seed)
    # Advance RNG to where we left off
    for _ in range(start_idx):
        make_freq_matched_permuted_values(bins, rng)

    log.info(f"Running frequency-matched null × {n_permutations:,} (bins={n_bins})")
    t0 = time.time()
    for i in range(start_idx, n_permutations):
        permuted_values = make_freq_matched_permuted_values(bins, rng)
        # Recompute abjad for this permutation
        abjad_i_p = np.array([compute_word_abjad(w, permuted_values) for w in word_i])
        abjad_j_p = np.array([compute_word_abjad(w, permuted_values) for w in word_j])
        abjad_dist_p = np.abs(abjad_i_p - abjad_j_p)

        rho_p = compute_partial_rho(abjad_dist_p, attentions, pos_distances, cooc_array)
        permuted_rhos.append(rho_p)

        if checkpoint_path and ((i + 1) % 10 == 0):
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "seed": seed,
                        "n_bins": n_bins,
                        "n_permutations": n_permutations,
                        "done": i + 1,
                        "observed_rho": float(observed_rho),
                        "permuted_rhos": permuted_rhos,
                    },
                    f,
                    ensure_ascii=False,
                )

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - start_idx) / max(elapsed, 1)
            eta = (n_permutations - i - 1) / max(rate, 1e-6)
            log.info(
                f"  permutation {i+1}/{n_permutations}: "
                f"latest ρ = {rho_p:.5f} | "
                f"{rate:.2f} perm/s | ETA {eta/60:.1f} min"
            )
            gc.collect()

    permuted_rhos_arr = np.array(permuted_rhos)
    abs_permuted = np.abs(permuted_rhos_arr)
    threshold_95 = np.percentile(abs_permuted, 95)
    n_more_extreme = int(np.sum(abs_permuted >= abs(observed_rho)))
    empirical_p = (n_more_extreme + 1) / (n_permutations + 1)
    passed = empirical_p < 0.05

    interpretation = (
        f"Observed |ρ| = {abs(observed_rho):.4f}; "
        f"95th-percentile of |permuted| (freq-matched, K={n_bins}) = {threshold_95:.4f}; "
        f"empirical p (two-tailed) = {empirical_p:.4f}. "
    )
    if passed:
        interpretation += (
            "PASS: the Mashriqi mapping outperforms random letter-value "
            "permutations even when restricted to within-frequency-tier "
            "shuffles. The Mashriqi-specific component has within-tier "
            "structure, not just frequency-tier alignment."
        )
    else:
        interpretation += (
            "FAIL: the Mashriqi mapping does not significantly outperform "
            "frequency-matched random permutations. The apparent specificity "
            "is primarily about how Mashriqi values align with letter-frequency "
            "tiers rather than within-tier letter-to-value structure."
        )

    return {
        "observed_rho": float(observed_rho),
        "n_permutations": n_permutations,
        "n_bins": n_bins,
        "permuted_mean": float(permuted_rhos_arr.mean()),
        "permuted_sd": float(permuted_rhos_arr.std()),
        "permuted_ci_95_pct": [
            float(np.percentile(permuted_rhos_arr, 2.5)),
            float(np.percentile(permuted_rhos_arr, 97.5)),
        ],
        "abs_threshold_95th_pct": float(threshold_95),
        "empirical_p_two_tailed": float(empirical_p),
        "passed": bool(passed),
        "interpretation": interpretation,
        "bins_composition": {
            f"bin_{i+1}": {
                "letters": b,
                "size": len(b),
                "freq_range": [
                    int(min(letter_freqs.get(l, 0) for l in b)),
                    int(max(letter_freqs.get(l, 0) for l in b)),
                ],
            }
            for i, b in enumerate(bins)
        },
    }


# ─────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Frequency-matched permutation null for Mashriqi Abjad-attention result"
    )
    parser.add_argument("--records", type=Path, required=True, help="Path to q2_records.pkl")
    parser.add_argument("--cooc", type=Path, required=True, help="Path to q2_cooc.pkl")
    parser.add_argument(
        "--existing-results",
        type=Path,
        required=False,
        help="Path to q2_results_with_robustness.json (for embedding new results into existing structure)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for the freq-matched null results",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000, help="Number of permutations"
    )
    parser.add_argument(
        "--bins", type=int, default=4, help="Number of frequency-matched bins (4 or 7 typically)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    log.info(f"Loading records from {args.records}")
    with open(args.records, "rb") as f:
        records = pickle.load(f)
    log.info(f"  {len(records):,} records loaded")

    log.info(f"Loading co-occurrence frequencies from {args.cooc}")
    with open(args.cooc, "rb") as f:
        cooc_freq = pickle.load(f)
    log.info(f"  {len(cooc_freq):,} pair frequencies loaded")

    checkpoint_path = args.output.with_suffix(".checkpoint.json")
    result = run_freq_matched_null(
        records=records,
        cooc_freq=cooc_freq,
        n_permutations=args.n_permutations,
        n_bins=args.bins,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
    )

    # Embed into existing-results structure if provided
    if args.existing_results and args.existing_results.exists():
        with open(args.existing_results, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing["results"]["freq_matched_null"] = result
        existing["results"]["freq_matched_null"]["meta"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "seed": args.seed,
            "n_bins": args.bins,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {"freq_matched_null": result, "meta": {"seed": args.seed, "n_bins": args.bins}},
                f,
                ensure_ascii=False,
                indent=2,
            )

    log.info(f"Wrote {args.output}")
    log.info("─" * 60)
    log.info(result["interpretation"])
    log.info("─" * 60)

    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info(f"Removed checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
