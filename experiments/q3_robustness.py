"""Robustness checks for Q3 Hebrew gematria/AlephBERT control results."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from q3_alephbert_control import (
    BASE_HEBREW_LETTERS,
    FINAL_TO_BASE,
    GEMATRIA_VALUES,
    generate_control_verdict,
    gematria_value,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("q3_robustness")


def make_permuted_mapping(rng: random.Random) -> dict[str, int]:
    base_values = [GEMATRIA_VALUES[ch] for ch in BASE_HEBREW_LETTERS]
    permuted_values = base_values.copy()
    rng.shuffle(permuted_values)
    mapping = dict(zip(BASE_HEBREW_LETTERS, permuted_values))
    for final, base in FINAL_TO_BASE.items():
        mapping[final] = mapping[base]
    return mapping


def partial_corr_spearman(
    records: list[dict],
    cooc_freq: dict,
    target: str = "gematria_distance",
    outcome: str = "mean_attention",
    control_vars: tuple[str, ...] = ("pos_distance", "cooc_freq"),
) -> dict:
    if len(records) < 50:
        return {"rho": None, "p": None, "n": len(records), "error": "insufficient data"}

    x_target = np.array([r[target] for r in records]).reshape(-1, 1)
    x_outcome = np.array([r[outcome] for r in records]).reshape(-1, 1)

    controls = []
    for var in control_vars:
        if var == "cooc_freq":
            vals = []
            for r in records:
                k1 = (r["word_i"], r["word_j"])
                k2 = (r["word_j"], r["word_i"])
                vals.append(cooc_freq.get(k1, cooc_freq.get(k2, 0.0)))
            controls.append(vals)
        else:
            controls.append([r[var] for r in records])

    x_controls = np.array(controls).T
    reg = LinearRegression().fit(x_controls, x_target.ravel())
    resid_target = x_target.ravel() - reg.predict(x_controls)
    reg2 = LinearRegression().fit(x_controls, x_outcome.ravel())
    resid_outcome = x_outcome.ravel() - reg2.predict(x_controls)
    rho, p = spearmanr(resid_target, resid_outcome)
    return {"rho": float(rho), "p": float(p), "n": len(records)}


def select_records(records: list[dict], group: str) -> list[dict]:
    if group == "all":
        return records
    if group == "different":
        return [r for r in records if r.get("root_relation") == "different"]
    if group == "same":
        return [r for r in records if r.get("root_relation") == "same"]
    raise ValueError(f"Unknown analysis group: {group}")


def permutation_null(
    records: list[dict],
    cooc_freq: dict,
    n_permutations: int,
    seed: int,
    analysis_group: str,
    checkpoint_path: str | None,
    checkpoint_interval: int,
) -> dict:
    rng = random.Random(seed)
    data = select_records(records, analysis_group)
    log.info(f"Running Hebrew gematria permutation null x {n_permutations:,}")
    log.info(f"  Analysis group: {analysis_group}; records: {len(data):,}")

    observed = partial_corr_spearman(data, cooc_freq)
    observed_rho = observed["rho"]
    log.info(f"  Observed rho = {observed_rho:.5f}")

    unique_words = set()
    for r in data:
        unique_words.add(r["word_i"])
        unique_words.add(r["word_j"])
    log.info(f"  Unique words: {len(unique_words):,}")

    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    permuted_rhos = []
    start_i = 0

    if checkpoint and checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            saved.get("seed") == seed
            and saved.get("n_permutations") == n_permutations
            and saved.get("analysis_group") == analysis_group
        ):
            permuted_rhos = [float(x) for x in saved.get("permuted_rhos", [])]
            start_i = len(permuted_rhos)
            for _ in range(start_i):
                make_permuted_mapping(rng)
            log.info(f"  Resuming from checkpoint: {start_i}/{n_permutations}")
        else:
            log.warning("  Ignoring checkpoint because run parameters differ")

    def write_checkpoint(done: int) -> None:
        if not checkpoint:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        payload = {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "n_permutations": n_permutations,
            "analysis_group": analysis_group,
            "done": done,
            "observed_rho": observed_rho,
            "observed_n": observed["n"],
            "permuted_rhos": permuted_rhos,
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(checkpoint)

    for i in range(start_i, n_permutations):
        mapping = make_permuted_mapping(rng)
        word_to_gematria = {w: gematria_value(w, mapping) for w in unique_words}
        permuted_records = []
        for r in data:
            permuted_records.append({
                **r,
                "gematria_distance": abs(
                    word_to_gematria[r["word_i"]] - word_to_gematria[r["word_j"]]
                ),
            })

        result = partial_corr_spearman(permuted_records, cooc_freq)
        permuted_rhos.append(result["rho"])
        del permuted_records

        if (i + 1) % checkpoint_interval == 0 or (i + 1) == n_permutations:
            write_checkpoint(i + 1)
            gc.collect()
        if (i + 1) % 100 == 0:
            log.info(f"  permutation {i+1}/{n_permutations}: latest rho = {result['rho']:.5f}")

    arr = np.array(permuted_rhos)
    n_extreme = int(np.sum(np.abs(arr) >= abs(observed_rho)))
    empirical_p = n_extreme / n_permutations
    abs_threshold_95 = float(np.percentile(np.abs(arr), 95.0))
    passed = abs(observed_rho) > abs_threshold_95

    return {
        "observed_rho": observed_rho,
        "observed_n": observed["n"],
        "analysis_group": analysis_group,
        "n_permutations": n_permutations,
        "permuted_mean": float(arr.mean()),
        "permuted_sd": float(arr.std(ddof=1)),
        "permuted_ci_95_pct": [
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        ],
        "empirical_p_two_tailed": empirical_p,
        "abs_threshold_95th_pct": abs_threshold_95,
        "passed": bool(passed),
        "interpretation": (
            f"Observed |rho| = {abs(observed_rho):.4f}; "
            f"95th-percentile of |permuted| = {abs_threshold_95:.4f}; "
            f"empirical p (two-tailed) = {empirical_p:.4f}. "
            f"{'PASS' if passed else 'FAIL'} for Hebrew gematria mapping specificity."
        ),
    }


def random_pair_cross_sentence_baseline(records: list[dict], cooc_freq: dict, seed: int, analysis_group: str) -> dict:
    rng = random.Random(seed)
    data = select_records(records, analysis_group)
    log.info("Running exact random-pair cross-sentence baseline")
    log.info(f"  Analysis group: {analysis_group}; records: {len(data):,}")

    observed = partial_corr_spearman(data, cooc_freq)
    observed_rho = observed["rho"]

    candidates = [(r["sentence_id"], r["word_j"]) for r in data if "sentence_id" in r]
    if not candidates:
        raise ValueError("Q3 records do not contain sentence_id; exact scrambling is unavailable")

    word_to_gematria = {}
    for r in data:
        word_to_gematria[r["word_i"]] = gematria_value(r["word_i"])
        word_to_gematria[r["word_j"]] = gematria_value(r["word_j"])

    pos_min = min(r["pos_distance"] for r in data)
    pos_max = max(r["pos_distance"] for r in data)
    cooc_mean = float(np.mean(list(cooc_freq.values()))) if cooc_freq else 0.0
    fake_cooc = _ConstCoocDict(cooc_mean)

    scrambled = []
    for r in data:
        new_j = None
        for _ in range(20):
            sent_id, candidate = rng.choice(candidates)
            if sent_id != r["sentence_id"]:
                new_j = candidate
                break
        if new_j is None:
            new_j = rng.choice(candidates)[1]

        scrambled.append({
            "word_i": r["word_i"],
            "word_j": new_j,
            "mean_attention": r["mean_attention"],
            "gematria_distance": abs(
                word_to_gematria.get(r["word_i"], 0) - word_to_gematria.get(new_j, 0)
            ),
            "pos_distance": rng.randint(pos_min, pos_max),
            "root_relation": "different",
        })

    scrambled_result = partial_corr_spearman(scrambled, fake_cooc)
    return {
        "observed_rho": observed_rho,
        "scrambled_rho": scrambled_result["rho"],
        "scrambled_p": scrambled_result["p"],
        "scrambled_n": scrambled_result["n"],
        "observed_minus_scrambled": (
            observed_rho - scrambled_result["rho"]
            if observed_rho is not None and scrambled_result["rho"] is not None
            else None
        ),
        "interpretation": (
            f"Observed rho = {observed_rho:.4f}; "
            f"random-pair scrambled rho = {scrambled_result['rho']:.4f}. "
            f"{'PASS' if abs(scrambled_result['rho']) < 0.5 * abs(observed_rho) else 'CONCERN'}: "
            "scrambled rho should be substantially smaller if the effect is pair-level."
        ),
    }


class _ConstCoocDict:
    def __init__(self, val):
        self._v = val

    def get(self, key, default=None):
        return self._v

    def values(self):
        return [self._v]


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness checks for Q3 Hebrew gematria control")
    parser.add_argument("--records", required=True)
    parser.add_argument("--cooc", required=True)
    parser.add_argument("--existing-results", default=None)
    parser.add_argument("--output", default="q3_results_with_robustness.json")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analysis-group", choices=["all", "different", "same"], default="all")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--skip-random-pair", action="store_true")
    args = parser.parse_args()

    log.info(f"Loading records from {args.records}")
    with open(args.records, "rb") as f:
        records = pickle.load(f)
    log.info(f"  {len(records):,} records loaded")

    log.info(f"Loading co-occurrence frequencies from {args.cooc}")
    with open(args.cooc, "rb") as f:
        cooc_freq = pickle.load(f)
    log.info(f"  {len(cooc_freq):,} pair frequencies loaded")

    robustness = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
            "n_records": len(records),
            "analysis_group": args.analysis_group,
        }
    }

    if not args.skip_permutation:
        checkpoint = args.checkpoint or str(Path(args.output).with_suffix(".checkpoint.json"))
        robustness["permutation_null"] = permutation_null(
            records,
            cooc_freq,
            n_permutations=args.n_permutations,
            seed=args.seed,
            analysis_group=args.analysis_group,
            checkpoint_path=checkpoint,
            checkpoint_interval=args.checkpoint_interval,
        )
        log.info(robustness["permutation_null"]["interpretation"])

    if not args.skip_random_pair:
        robustness["random_pair_baseline"] = random_pair_cross_sentence_baseline(
            records, cooc_freq, seed=args.seed, analysis_group=args.analysis_group
        )
        log.info(robustness["random_pair_baseline"]["interpretation"])

    if args.existing_results and Path(args.existing_results).exists():
        with open(args.existing_results, "r", encoding="utf-8") as f:
            existing = json.load(f)
        results = existing.get("results", {})
        if "primary" not in results and "primary_all_pairs" in results:
            results["primary"] = results["primary_all_pairs"]
        results.update(robustness)
        out = {"results": results, "verdict": generate_control_verdict(results)}
    else:
        out = {"robustness": robustness}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
