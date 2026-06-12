"""
AL-MIR'ĀH — Q1 Real-Model Runner
===================================
Runs the root-cluster density experiment using real embedding models:
  Arabic  → Twitter AraVec (tweets_cbow_300) or full_grams_cbow_300
  English → word2vec-google-news-300 (gensim cached)

Usage:
  # Twitter AraVec run:
  venv/bin/python q1_experiment/run_q1_real.py \\
      --arabic /path/to/twitter_cbow_300.bin \\
      --run-name twitter

  # Full grams run:
  venv/bin/python q1_experiment/run_q1_real.py \\
      --arabic /path/to/full_grams_cbow_300.bin \\
      --run-name full

Results are written to q1_experiment/results/<run-name>/
"""

import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from q1_root_cluster_density import (
    ARABIC_ROOT_FAMILIES,
    ENGLISH_MORPH_FAMILIES,
    compute_root_cluster_density,
    print_comparative_summary,
    generate_visualizations,
    save_results,
    strip_diacritics,
)


def main():
    parser = argparse.ArgumentParser(description="Q1 real-model experiment runner")
    parser.add_argument("--arabic", required=True,
                        help="Path to AraVec .bin file (word2vec binary format)")
    parser.add_argument("--run-name", default="twitter",
                        choices=["twitter", "full"],
                        help="Name for this run (sets output subdirectory)")
    parser.add_argument("--english", default="download",
                        help="English model: 'download' (use cached gensim) or path to .bin")
    parser.add_argument("--min-words", type=int, default=3,
                        help="Min words per family to include (default: 3)")
    args = parser.parse_args()

    try:
        from gensim.models import KeyedVectors
        import gensim.downloader as gensim_api
    except ImportError:
        print("ERROR: gensim not installed. Run: pip install gensim")
        sys.exit(1)

    # ── Load Arabic model ──────────────────────────────────────────────
    print(f"\nLoading Arabic model: {args.arabic}")
    try:
        arabic_model = KeyedVectors.load_word2vec_format(args.arabic, binary=True)
        print(f"  ✓ Loaded. Vocabulary size: {len(arabic_model.key_to_index):,}")
    except Exception as e:
        print(f"  ✗ Failed to load Arabic model: {e}")
        sys.exit(1)

    # Quick vocab probe — show coverage before running
    print("\n  Vocabulary probe (undiacritised lookup):")
    total_found = 0
    total_words = 0
    for root, words in ARABIC_ROOT_FAMILIES.items():
        found = [strip_diacritics(w) for w in words
                 if strip_diacritics(w) in arabic_model.key_to_index]
        total_found += len(found)
        total_words += len(words)
        pct = 100.0 * len(found) / len(words)
        print(f"    {root}: {len(found)}/{len(words)} ({pct:.0f}%)")
    print(f"  Overall Arabic coverage: {total_found}/{total_words} "
          f"({100.0*total_found/total_words:.1f}%)")

    # ── Load English model ─────────────────────────────────────────────
    print(f"\nLoading English model: {args.english}")
    try:
        if args.english == "download":
            print("  Using cached word2vec-google-news-300...")
            english_model = gensim_api.load("word2vec-google-news-300")
        else:
            english_model = KeyedVectors.load_word2vec_format(args.english, binary=True)
        print(f"  ✓ Loaded. Vocabulary size: {len(english_model.key_to_index):,}")
    except Exception as e:
        print(f"  ✗ Failed to load English model: {e}")
        sys.exit(1)

    # ── Output directory ───────────────────────────────────────────────
    out_dir = os.path.join(HERE, "results", args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Run experiment ─────────────────────────────────────────────────
    arabic_results = compute_root_cluster_density(
        arabic_model, ARABIC_ROOT_FAMILIES,
        f"Arabic (AraVec {args.run_name})", args.min_words, arabic=True
    )
    english_results = compute_root_cluster_density(
        english_model, ENGLISH_MORPH_FAMILIES,
        "English (word2vec-google-news-300)", args.min_words, arabic=False
    )

    # ── Summary ────────────────────────────────────────────────────────
    print_comparative_summary(arabic_results, english_results)

    # ── Visualizations + JSON ─────────────────────────────────────────
    print(f"\nGenerating visualizations → {out_dir}/")
    generate_visualizations(arabic_results, english_results, out_dir)

    print(f"\nSaving results → {out_dir}/")
    save_results(arabic_results, english_results, out_dir)

    # ── Final summary block (Task 4 requirement) ───────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY — {args.run_name.upper()} RUN")
    print(f"{'='*60}")
    if arabic_results:
        ar = arabic_results
        print(f"\n  Arabic (AraVec {args.run_name}):")
        print(f"    Gap (intra - cross):  {ar['gap']:.4f}")
        print(f"    Cohen's d:            {ar['cohens_d']:.4f}")
        print(f"    p-value:              {ar['p_value']:.3e}")
        print(f"    Vocab coverage:       {ar['vocab_coverage_pct']:.1f}%")
        print(f"\n    Per-root coverage:")
        for root, stats in ar['family_stats'].items():
            print(f"      {root}: {stats['words_available']}/{stats['words_total']} "
                  f"({stats['coverage_pct']:.0f}%)")
    if english_results:
        en = english_results
        print(f"\n  English (word2vec):")
        print(f"    Gap (intra - cross):  {en['gap']:.4f}")
        print(f"    Cohen's d:            {en['cohens_d']:.4f}")
        print(f"    p-value:              {en['p_value']:.3e}")
        print(f"    Vocab coverage:       {en['vocab_coverage_pct']:.1f}%")
    if arabic_results and english_results:
        ratio = arabic_results['gap'] / english_results['gap'] if english_results['gap'] > 0 else float('inf')
        print(f"\n  Arabic/English gap ratio: {ratio:.2f}x")
        claim = arabic_results['gap'] > english_results['gap'] and arabic_results['significant']
        print(f"  Claim supported: {'YES ✓' if claim else 'NO ✗'}")
    print(f"\n  Results saved to: {out_dir}/q1_results.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
