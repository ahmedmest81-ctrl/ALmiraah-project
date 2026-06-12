"""
AL-MIR'AH - Q3: Hebrew Gematria Control with AlephBERT
======================================================

Control question: does a Hebrew gematria distance signal appear in AlephBERT
attention over Hebrew-script classical Jewish texts, using the same
operational logic as Q2?

This script is adapted from the patched Q2 pipeline but is deliberately
Hebrew-specific:
- model: onlplab/alephbert-base
- corpus: cleaned Sefaria Hebrew text, one passage/sentence per line
- mapping: standard Hebrew gematria, including final letter forms
- records/co-occurrence artifacts are saved for robustness checks
- sentence_id is saved per record so cross-sentence scrambling is exact
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("q3")


ALEPHBERT_MODEL = "onlplab/alephbert-base"
RANDOM_SEED = 42
WORKING_SEQ_LEN = 128
BATCH_SIZE = 8
N_SENTENCES = 100000
MIN_SENTENCE_TOKENS = 6
MAX_SENTENCE_TOKENS = 80

GEMATRIA_VALUES = {
    "א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9,
    "י": 10, "כ": 20, "ך": 20, "ל": 30, "מ": 40, "ם": 40, "נ": 50, "ן": 50,
    "ס": 60, "ע": 70, "פ": 80, "ף": 80, "צ": 90, "ץ": 90, "ק": 100,
    "ר": 200, "ש": 300, "ת": 400,
}

BASE_HEBREW_LETTERS = [
    "א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "י", "כ", "ל", "מ", "נ",
    "ס", "ע", "פ", "צ", "ק", "ר", "ש", "ת",
]

FINAL_TO_BASE = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
HEBREW_RE = re.compile(r"[\u0590-\u05FF]+")
NIQQUD_RE = re.compile(r"[\u0591-\u05C7]")


def strip_hebrew_marks(text: str) -> str:
    return NIQQUD_RE.sub("", text)


def clean_hebrew(text: str) -> str:
    """Keep Hebrew letters, remove cantillation/niqqud/HTML, collapse spaces."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = strip_hebrew_marks(text)
    text = re.sub(r"[^\u05D0-\u05EA\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def gematria_value(word: str, mapping: dict[str, int] | None = None) -> int:
    mapping = GEMATRIA_VALUES if mapping is None else mapping
    return sum(mapping.get(ch, 0) for ch in word)


HEBREW_PREFIXES = [
    "וכשה", "וכש", "שה", "כש", "וה", "וב", "ול", "ומ", "וש", "ה", "ו", "ב",
    "ל", "מ", "כ", "ש",
]
HEBREW_SUFFIXES = ["יהם", "יהן", "יכם", "יכן", "ינו", "ות", "ים", "ין", "ה", "ו", "י", "ך"]
WEAK_HEBREW = {"א", "ה", "ו", "י"}


def hebrew_root_heuristic(word: str) -> str:
    """
    Very conservative root proxy for exploratory controls only.

    Hebrew morphology is not safely recoverable with this heuristic. The
    primary Q3 result therefore uses all analyzable pairs. This function only
    enables a rough same-root/different-root breakdown.
    """
    w = clean_hebrew(word)
    if len(w) < 3:
        return ""

    for prefix in HEBREW_PREFIXES:
        if w.startswith(prefix) and len(w) > len(prefix) + 2:
            w = w[len(prefix):]
            break

    for suffix in HEBREW_SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            w = w[:-len(suffix)]
            break

    consonants = [ch for ch in w if ch in GEMATRIA_VALUES and ch not in WEAK_HEBREW]
    if len(consonants) < 3:
        return ""
    return "".join(consonants[:3])


def same_root_heuristic(w1: str, w2: str) -> bool | None:
    r1 = hebrew_root_heuristic(w1)
    r2 = hebrew_root_heuristic(w2)
    if not r1 or not r2:
        return None
    return r1 == r2


def load_corpus_from_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    raw_sentences = re.split(r"[.\n!?;:׃]+", text)
    sentences = []
    for s in raw_sentences:
        s = clean_hebrew(s)
        tokens = s.split()
        if MIN_SENTENCE_TOKENS <= len(tokens) <= MAX_SENTENCE_TOKENS:
            sentences.append(s)
    log.info(f"Loaded {len(sentences):,} Hebrew passages from {path}")
    return sentences


class AttentionExtractor:
    def __init__(self, model_name: str = ALEPHBERT_MODEL, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_attentions=True,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        log.info("Model loaded.")

    def extract(self, sentences: list[str], n_sentences: int) -> list[dict]:
        import torch

        all_records = []
        total = min(len(sentences), n_sentences)

        for sentence_id, sentence in enumerate(sentences[:total]):
            if sentence_id % 500 == 0:
                log.info(f"  Sentence {sentence_id}/{total}...")

            words = sentence.split()
            if not (MIN_SENTENCE_TOKENS <= len(words) <= MAX_SENTENCE_TOKENS):
                continue

            encoded = self.tokenizer(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                max_length=WORKING_SEQ_LEN,
                truncation=True,
                padding=False,
            ).to(self.device)

            if encoded["input_ids"].shape[1] < 4:
                continue

            with torch.no_grad():
                outputs = self.model(**encoded)

            attn = torch.stack(outputs.attentions, dim=0).squeeze(1)
            mean_attn = attn.mean(dim=(0, 1)).cpu().numpy()

            word_positions = {}
            for tok_idx, word_idx in enumerate(encoded.word_ids()):
                if word_idx is not None and word_idx not in word_positions:
                    word_positions[word_idx] = tok_idx

            n_words = len(words)
            for wi in range(n_words):
                for wj in range(wi + 1, n_words):
                    if wi not in word_positions or wj not in word_positions:
                        continue

                    ti = word_positions[wi]
                    tj = word_positions[wj]
                    if ti >= mean_attn.shape[0] or tj >= mean_attn.shape[0]:
                        continue

                    word_i = words[wi]
                    word_j = words[wj]
                    gem_i = gematria_value(word_i)
                    gem_j = gematria_value(word_j)
                    if gem_i == 0 or gem_j == 0:
                        continue

                    sr = same_root_heuristic(word_i, word_j)
                    root_rel = "unknown" if sr is None else ("same" if sr else "different")

                    all_records.append({
                        "sentence_id": sentence_id,
                        "word_i": word_i,
                        "word_j": word_j,
                        "pos_i": wi,
                        "pos_j": wj,
                        "pos_distance": wj - wi,
                        "gematria_i": gem_i,
                        "gematria_j": gem_j,
                        "gematria_distance": abs(gem_i - gem_j),
                        "mean_attention": float((mean_attn[ti, tj] + mean_attn[tj, ti]) / 2),
                        "root_relation": root_rel,
                    })

        log.info(f"Extracted {len(all_records):,} Hebrew word-pair attention records")
        return all_records


def compute_cooccurrence_frequency(sentences: list[str], vocab: set[str]) -> dict:
    log.info("Computing co-occurrence frequencies...")
    window = 40
    step = 20
    all_tokens = " ".join(sentences).split()

    passages = []
    for start in range(0, max(0, len(all_tokens) - window), step):
        passages.append(set(all_tokens[start:start + window]) & vocab)
    log.info(f"  {len(passages):,} passages")

    if not passages:
        return {}

    word_freq = Counter(w for p in passages for w in p)
    n_passages = len(passages)
    cooc = defaultdict(int)
    for passage in passages:
        words_list = sorted(passage)
        for i, w1 in enumerate(words_list):
            for w2 in words_list[i + 1:]:
                cooc[(w1, w2)] += 1

    freq_map = {}
    for (w1, w2), count in cooc.items():
        if count >= 2:
            p_w1 = word_freq[w1] / n_passages
            p_w2 = word_freq[w2] / n_passages
            p_joint = count / n_passages
            val = p_joint / (p_w1 * p_w2 + 1e-10)
            freq_map[(w1, w2)] = val
            freq_map[(w2, w1)] = val
    log.info(f"  {len(freq_map):,} co-occurrence pairs computed")
    return freq_map


def partial_corr_spearman(
    data: list[dict],
    cooc_freq: dict,
    target: str = "gematria_distance",
    outcome: str = "mean_attention",
    control_vars: tuple[str, ...] = ("pos_distance", "cooc_freq"),
) -> dict:
    if len(data) < 50:
        return {"rho": None, "p": None, "n": len(data), "error": "insufficient data"}

    from sklearn.linear_model import LinearRegression

    x_target = np.array([r[target] for r in data]).reshape(-1, 1)
    x_outcome = np.array([r[outcome] for r in data]).reshape(-1, 1)

    controls = []
    for var in control_vars:
        if var == "cooc_freq":
            vals = []
            for r in data:
                k1 = (r["word_i"], r["word_j"])
                k2 = (r["word_j"], r["word_i"])
                vals.append(cooc_freq.get(k1, cooc_freq.get(k2, 0.0)))
            controls.append(vals)
        else:
            controls.append([r[var] for r in data])

    x_controls = np.array(controls).T
    reg = LinearRegression().fit(x_controls, x_target.ravel())
    resid_target = x_target.ravel() - reg.predict(x_controls)
    reg2 = LinearRegression().fit(x_controls, x_outcome.ravel())
    resid_outcome = x_outcome.ravel() - reg2.predict(x_controls)

    rho, p = spearmanr(resid_target, resid_outcome)
    return {"rho": float(rho), "p": float(p), "n": len(data)}


def run_analysis(records: list[dict], cooc_freq: dict) -> dict:
    log.info("Running Q3 partial correlation analysis...")
    all_pairs = records
    different = [r for r in records if r["root_relation"] == "different"]
    same = [r for r in records if r["root_relation"] == "same"]
    unknown = [r for r in records if r["root_relation"] == "unknown"]
    log.info(f"  All analyzable pairs: {len(all_pairs):,}")
    log.info(f"  Heuristic different-root pairs: {len(different):,}")
    log.info(f"  Heuristic same-root pairs: {len(same):,}")
    log.info(f"  Heuristic unknown-root pairs: {len(unknown):,}")

    results = {
        "primary_all_pairs": partial_corr_spearman(all_pairs, cooc_freq),
        "heuristic_different_root": partial_corr_spearman(different, cooc_freq),
        "heuristic_same_root": partial_corr_spearman(same, cooc_freq),
    }
    # Keep a Q2-compatible key for downstream tables while preserving the more
    # explicit Q3 label. For Hebrew, the primary result is all analyzable pairs;
    # the root split is heuristic/exploratory.
    results["primary"] = results["primary_all_pairs"]

    primary = results["primary_all_pairs"]
    if primary.get("rho") is not None:
        log.info(
            f"  Primary all-pairs result: rho={primary['rho']:.4f}, "
            f"p={primary['p']:.4e}, n={primary['n']:,}"
        )

    bins = [
        ("proximate_0_10", lambda d: d <= 10),
        ("medial_11_25", lambda d: 11 <= d <= 25),
        ("medial_26_50", lambda d: 26 <= d <= 50),
        ("distal_51_100", lambda d: 51 <= d <= 100),
        ("distal_101_200", lambda d: 101 <= d <= 200),
        ("distal_201_plus", lambda d: d > 200),
    ]
    results["binned"] = {}
    for name, pred in bins:
        subset = [r for r in all_pairs if pred(r["gematria_distance"])]
        if subset:
            attns = [r["mean_attention"] for r in subset]
            results["binned"][name] = {
                "n": len(subset),
                "mean_attention": float(np.mean(attns)),
                "median_attention": float(np.median(attns)),
                "sd": float(np.std(attns)),
            }
        else:
            results["binned"][name] = {"n": 0}

    if all_pairs:
        rho_raw, p_raw = spearmanr(
            [r["gematria_distance"] for r in all_pairs],
            [r["mean_attention"] for r in all_pairs],
        )
        results["unadjusted_all_pairs"] = {
            "rho": float(rho_raw),
            "p": float(p_raw),
            "n": len(all_pairs),
        }

    return results


def generate_control_verdict(results: dict) -> str:
    primary = results.get("primary_all_pairs", {})
    rho = primary.get("rho")
    p = primary.get("p")
    n = primary.get("n")
    if rho is None or p is None or n is None:
        return "VERDICT: Insufficient data to evaluate Q3 Hebrew control."

    if p < 0.05 and rho < 0:
        return (
            "VERDICT: HEBREW CONTROL SIGNAL DETECTED - gematria proximity predicts "
            f"higher AlephBERT mutual attention (partial rho = {rho:.4f}, "
            f"p = {p:.4e}, n = {n:,}) after controlling for positional distance "
            "and co-occurrence frequency. This should be interpreted as a control "
            "signal, not as direct support for the Arabic Abjad claim."
        )
    if p < 0.05 and rho > 0:
        return (
            "VERDICT: HEBREW CONTROL OPPOSITE-DIRECTION SIGNAL - gematria distance "
            f"predicts higher attention (partial rho = {rho:.4f}, p = {p:.4e}, "
            f"n = {n:,})."
        )
    return (
        "VERDICT: HEBREW CONTROL NULL - no significant partial correlation between "
        f"gematria distance and AlephBERT attention (rho = {rho:.4f}, "
        f"p = {p:.4e}, n = {n:,})."
    )


def save_results(results: dict, verdict: str, output_dir: str, n_sentences: int) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {"results": results, "verdict": verdict}
    with (out_dir / "q3_results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info(f"Results written to {out_dir / 'q3_results.json'}")

    primary = results.get("primary_all_pairs", {})
    with (out_dir / "q3_report.txt").open("w", encoding="utf-8") as f:
        f.write("AL-MIR'AH - Q3: Hebrew Gematria Control with AlephBERT\n")
        f.write("=" * 64 + "\n\n")
        f.write(f"Model: {ALEPHBERT_MODEL}\n")
        f.write("Corpus: Sefaria Hebrew-script classical corpus (Hebrew/Aramaic)\n")
        f.write(f"Sentences sampled: {n_sentences}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")
        f.write("PRIMARY RESULT (all analyzable pairs)\n")
        f.write("-" * 40 + "\n")
        f.write("Partial rho (gematria distance ~ attention, controlling pos_dist + cooc_freq):\n")
        f.write(f"  rho = {primary.get('rho', 'N/A')}\n")
        f.write(f"  p = {primary.get('p', 'N/A')}\n")
        f.write(f"  n = {primary.get('n', 0):,}\n\n")
        f.write("HEURISTIC ROOT BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for key in ["heuristic_different_root", "heuristic_same_root"]:
            r = results.get(key, {})
            f.write(f"  {key}: rho={r.get('rho')}, p={r.get('p')}, n={r.get('n', 0):,}\n")
        f.write("\nBINNED ANALYSIS\n")
        f.write("-" * 40 + "\n")
        for label, stats in results.get("binned", {}).items():
            if stats.get("n", 0):
                f.write(
                    f"  {label}: n={stats['n']:,}, "
                    f"mean_attn={stats['mean_attention']:.6f}, "
                    f"median={stats['median_attention']:.6f}\n"
                )
        f.write("\nVERDICT\n")
        f.write("-" * 40 + "\n")
        f.write(verdict + "\n")
    log.info(f"Report written to {out_dir / 'q3_report.txt'}")
    print("\n" + "=" * 64)
    print(verdict)
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q3: Hebrew Gematria x AlephBERT attention control")
    parser.add_argument("--corpus", type=str, default="corpus_hebrew.txt")
    parser.add_argument("--n-sentences", type=int, default=N_SENTENCES)
    parser.add_argument("--output", type=str, default="q3_results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    sentences = load_corpus_from_file(args.corpus)
    random.shuffle(sentences)
    sentences = sentences[:args.n_sentences]
    log.info(f"Using {len(sentences):,} Hebrew passages")

    vocab = set()
    for sentence in sentences:
        vocab.update(sentence.split())
    log.info(f"Vocabulary: {len(vocab):,} types")

    cooc_freq = compute_cooccurrence_frequency(sentences, vocab)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "q3_cooc.pkl").open("wb") as f:
        pickle.dump(cooc_freq, f)
    log.info(f"Saved co-occurrence dict to {out_dir / 'q3_cooc.pkl'}")

    extractor = AttentionExtractor(device=args.device)
    records = extractor.extract(sentences, n_sentences=args.n_sentences)
    if not records:
        log.error("No records extracted. Check corpus format.")
        return

    with (out_dir / "q3_records.pkl").open("wb") as f:
        pickle.dump(records, f)
    log.info(f"Saved records to {out_dir / 'q3_records.pkl'} ({len(records):,} pairs)")

    results = run_analysis(records, cooc_freq)
    verdict = generate_control_verdict(results)
    save_results(results, verdict, args.output, len(sentences))


if __name__ == "__main__":
    main()
