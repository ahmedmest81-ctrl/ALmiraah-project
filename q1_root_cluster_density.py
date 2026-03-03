"""
AL-MIR'ĀH Research Framework
Q1 Proof of Concept: Root-Cluster Density
==========================================

Measurement 1: Do Arabic words sharing a triconsonantal root cluster
more tightly in embedding space than English words sharing a prefix/suffix?

Prediction (from morphology_research_spec.docx):
  Arabic intra-root cosine similarity > cross-root cosine similarity
  This gap is LARGER in Arabic than in English
  Both AraVec and CAMeLBERT show similar patterns

Models required (download separately):
  Arabic: AraVec - https://github.com/bakrianoo/aravec
          Use: full_grams_cbow_300 (recommended) or tweets model
  English: word2vec Google News vectors
           https://code.google.com/archive/p/word2vec/
           OR use gensim's downloader: gensim.downloader.load('word2vec-google-news-300')

Usage:
  pip install gensim numpy scipy scikit-learn matplotlib seaborn
  python q1_root_cluster_density.py --arabic /path/to/aravec.bin --english /path/to/GoogleNews.bin

Results are saved to: q1_results/
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Arabic root lexicon ───────────────────────────────────────────────────────
# Hand-curated root families: each entry is (root, [words_in_that_root_family])
# These are high-frequency roots with multiple attested forms in AraVec corpus
# Source: Lane's Lexicon + framework Pillar II lexicon
ARABIC_ROOT_FAMILIES = {
    "ك-ت-ب": [
        "كَتَبَ", "كِتَاب", "كَاتِب", "مَكْتُوب", "كِتَابَة", "مَكْتَب",
        "كُتُب", "كَتَّبَ", "اكْتَتَبَ", "كَتَبَة"
    ],
    "ع-ل-م": [
        "عَلِمَ", "عِلْم", "عَالِم", "مَعْلُوم", "تَعَلَّمَ", "عَلَّمَ",
        "مُعَلِّم", "مُتَعَلِّم", "عُلُوم", "مَعْلُومَات"
    ],
    "د-ر-س": [
        "دَرَسَ", "دِرَاسَة", "دَارِس", "مَدْرَسَة", "دَرَّسَ", "مُدَرِّس",
        "دُرُوس", "مَدْرُوس", "دِرَاسِي", "اسْتَدْرَسَ"
    ],
    "ق-ر-أ": [
        "قَرَأَ", "قِرَاءَة", "قَارِئ", "مَقْرُوء", "قَرَّأَ", "اقْتَرَأَ",
        "قُرَّاء", "مِقْرَاء", "تَقْرِيء", "قِرَاءَات"
    ],
    "خ-ر-ج": [
        "خَرَجَ", "خُرُوج", "خَارِج", "مَخْرَج", "أَخْرَجَ", "تَخَرَّجَ",
        "اسْتَخْرَجَ", "خِرِّيج", "خَرَّجَ", "مُخْرِج"
    ],
    "ر-ج-ع": [
        "رَجَعَ", "رُجُوع", "رَاجِع", "مَرْجِع", "أَرْجَعَ", "تَرَاجَعَ",
        "اسْتَرْجَعَ", "رَجْعَة", "مَرْجِعِي", "رَاجَعَ"
    ],
    "ح-م-ل": [
        "حَمَلَ", "حَمْل", "حَامِل", "مَحْمُول", "أَحْمَلَ", "تَحَمَّلَ",
        "احْتَمَلَ", "حِمْل", "حَمَّالَة", "مُحَمَّل"
    ],
    "ف-ت-ح": [
        "فَتَحَ", "فَتْح", "فَاتِح", "مَفْتُوح", "فَتَّحَ", "انْفَتَحَ",
        "افْتَتَحَ", "فِتَاح", "مَفْتَاح", "فَتَّاح"
    ],
    "ص-ل-ح": [
        "صَلَحَ", "صَلَاح", "صَالِح", "مَصْلَحَة", "أَصْلَحَ", "اسْتَصْلَحَ",
        "إِصْلَاح", "مُصْلِح", "صَلُوح", "تَصَالَحَ"
    ],
    "ق-و-ل": [
        "قَالَ", "قَوْل", "قَائِل", "مَقُول", "أَقْوَال", "قَوَّلَ",
        "تَقَوَّلَ", "مَقَال", "قِيل", "مَقُولَة"
    ],
    "ع-م-ل": [
        "عَمِلَ", "عَمَل", "عَامِل", "مَعْمُول", "أَعْمَال", "عَمَّلَ",
        "اعْتَمَلَ", "عَمَلِي", "مَعْمَل", "عُمَّال"
    ],
    "ن-ظ-ر": [
        "نَظَرَ", "نَظَر", "نَاظِر", "مَنْظُور", "أَنْظَار", "نَظَّرَ",
        "تَنَاظَرَ", "مِنْظَار", "نَظَرِي", "مَنْظَر"
    ],
    "ب-ح-ث": [
        "بَحَثَ", "بَحْث", "بَاحِث", "مَبْحُوث", "أَبْحَاث", "بَحَّثَ",
        "ابْتَحَثَ", "بَحْثِي", "مَبْحَث", "بُحُوث"
    ],
    "س-م-ع": [
        "سَمِعَ", "سَمَاع", "سَامِع", "مَسْمُوع", "أَسْمَعَ", "اسْتَمَعَ",
        "تَسَامَعَ", "سَمِيع", "مِسْمَع", "سَمَّاعَة"
    ],
    "ج-ل-س": [
        "جَلَسَ", "جُلُوس", "جَالِس", "مَجْلِس", "أَجْلَسَ", "جَلَّسَ",
        "اجْتَلَسَ", "جِلْسَة", "مُجَالِس", "جُلَسَاء"
    ],
}

# ── English root/morphological families (prefix/suffix clusters for comparison)
# English does NOT have templatic morphology — these are morphological families
# but structurally different: stem + affixes, not root + pattern
ENGLISH_MORPH_FAMILIES = {
    "write": [
        "write", "writes", "wrote", "written", "writing", "writer",
        "writers", "rewrite", "overwrite", "handwriting"
    ],
    "know": [
        "know", "knows", "knew", "known", "knowing", "knowledge",
        "knowledgeable", "unknown", "foreknowledge", "knower"
    ],
    "study": [
        "study", "studies", "studied", "studying", "student",
        "students", "studious", "studio", "understudy", "coursestudy"
    ],
    "read": [
        "read", "reads", "reading", "reader", "readers", "readout",
        "readable", "readability", "readership", "misread"
    ],
    "work": [
        "work", "works", "worked", "working", "worker", "workers",
        "workplace", "workout", "overwork", "teamwork"
    ],
    "speak": [
        "speak", "speaks", "spoke", "spoken", "speaking", "speaker",
        "speakers", "outspoken", "spokesperson", "speech"
    ],
    "teach": [
        "teach", "teaches", "taught", "teaching", "teacher", "teachers",
        "reteach", "overteach", "teachable", "teachings"
    ],
    "build": [
        "build", "builds", "built", "building", "builder", "builders",
        "rebuild", "buildings", "buildout", "groundbreaking"
    ],
    "open": [
        "open", "opens", "opened", "opening", "opener", "openly",
        "openness", "reopen", "openings", "openhanded"
    ],
    "see": [
        "see", "sees", "saw", "seen", "seeing", "seer",
        "foresee", "oversee", "oversight", "unseeing"
    ],
}


# ── Core analysis functions ───────────────────────────────────────────────────

def get_available_words(model, word_list):
    """Return words from list that exist in the model vocabulary."""
    available = []
    for word in word_list:
        try:
            _ = model[word]
            available.append(word)
        except KeyError:
            pass
    return available


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def compute_pairwise_similarities(model, word_list):
    """Compute all pairwise cosine similarities within a word list."""
    vectors = [model[w] for w in word_list]
    similarities = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_similarity(vectors[i], vectors[j])
            similarities.append(sim)
    return similarities


def compute_root_cluster_density(model, families, lang_name, min_words=3):
    """
    Measurement 1: Root-Cluster Density

    For each root/morphological family:
      - Compute intra-family pairwise cosine similarities
    Then compute cross-family similarities (random sample)

    Returns dict with all statistics.
    """
    print(f"\n{'='*60}")
    print(f"MEASUREMENT 1: ROOT-CLUSTER DENSITY — {lang_name}")
    print(f"{'='*60}")

    intra_sims_all = []
    family_stats = {}
    available_families = {}

    # ── Intra-family similarities ─────────────────────────────────────
    for root, words in families.items():
        available = get_available_words(model, words)
        if len(available) < min_words:
            print(f"  SKIP {root}: only {len(available)}/{len(words)} words in vocab")
            continue

        available_families[root] = available
        sims = compute_pairwise_similarities(model, available)
        intra_sims_all.extend(sims)

        family_stats[root] = {
            "words_available": len(available),
            "words_total": len(words),
            "intra_mean": float(np.mean(sims)),
            "intra_std": float(np.std(sims)),
            "intra_min": float(np.min(sims)),
            "intra_max": float(np.max(sims)),
            "available_words": available,
        }
        print(f"  {root}: {len(available)}/{len(words)} words | "
              f"intra-sim = {np.mean(sims):.4f} ± {np.std(sims):.4f}")

    if not available_families:
        print(f"  ERROR: No families had enough words in vocabulary.")
        return None

    # ── Cross-family similarities ─────────────────────────────────────
    print(f"\n  Computing cross-family similarities...")
    cross_sims_all = []
    family_list = list(available_families.keys())
    n_cross_samples = min(1000, len(intra_sims_all) * 3)

    rng = np.random.default_rng(42)
    for _ in range(n_cross_samples):
        # Pick two different families
        f1, f2 = rng.choice(len(family_list), size=2, replace=False)
        root1 = family_list[f1]
        root2 = family_list[f2]
        # Pick one word from each
        w1 = rng.choice(available_families[root1])
        w2 = rng.choice(available_families[root2])
        sim = cosine_similarity(model[w1], model[w2])
        cross_sims_all.append(sim)

    # ── Statistics ────────────────────────────────────────────────────
    intra_mean = float(np.mean(intra_sims_all))
    intra_std = float(np.std(intra_sims_all))
    cross_mean = float(np.mean(cross_sims_all))
    cross_std = float(np.std(cross_sims_all))
    gap = intra_mean - cross_mean

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(intra_sims_all)**2 + np.std(cross_sims_all)**2) / 2)
    cohens_d = gap / pooled_std if pooled_std > 0 else 0.0

    # Statistical significance (Mann-Whitney U — non-parametric)
    from scipy import stats
    u_stat, p_value = stats.mannwhitneyu(
        intra_sims_all, cross_sims_all, alternative='greater'
    )

    print(f"\n  ── RESULTS: {lang_name} ──")
    print(f"  Families analyzed:     {len(available_families)}")
    print(f"  Intra-family pairs:    {len(intra_sims_all)}")
    print(f"  Cross-family pairs:    {len(cross_sims_all)}")
    print(f"  Intra-family mean sim: {intra_mean:.4f} ± {intra_std:.4f}")
    print(f"  Cross-family mean sim: {cross_mean:.4f} ± {cross_std:.4f}")
    print(f"  GAP (intra - cross):   {gap:.4f}")
    print(f"  Cohen's d:             {cohens_d:.4f}")
    print(f"  Mann-Whitney p-value:  {p_value:.6f}")
    print(f"  Significant (p<0.05):  {'YES ✓' if p_value < 0.05 else 'NO ✗'}")

    return {
        "language": lang_name,
        "n_families": len(available_families),
        "n_intra_pairs": len(intra_sims_all),
        "n_cross_pairs": len(cross_sims_all),
        "intra_mean": intra_mean,
        "intra_std": intra_std,
        "cross_mean": cross_mean,
        "cross_std": cross_std,
        "gap": gap,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "family_stats": family_stats,
        "intra_sims_sample": intra_sims_all[:200],
        "cross_sims_sample": cross_sims_all[:200],
    }


def print_comparative_summary(arabic_results, english_results):
    """Print the head-to-head comparison — the core result."""
    print(f"\n{'='*60}")
    print("COMPARATIVE SUMMARY — THE KEY RESULT")
    print(f"{'='*60}")

    if arabic_results and english_results:
        ar_gap = arabic_results["gap"]
        en_gap = english_results["gap"]
        gap_ratio = ar_gap / en_gap if en_gap > 0 else float('inf')

        print(f"\n  {'Metric':<35} {'Arabic':>12} {'English':>12}")
        print(f"  {'-'*60}")
        print(f"  {'Intra-family mean similarity':<35} "
              f"{arabic_results['intra_mean']:>12.4f} "
              f"{english_results['intra_mean']:>12.4f}")
        print(f"  {'Cross-family mean similarity':<35} "
              f"{arabic_results['cross_mean']:>12.4f} "
              f"{english_results['cross_mean']:>12.4f}")
        print(f"  {'GAP (intra - cross)':<35} "
              f"{ar_gap:>12.4f} "
              f"{en_gap:>12.4f}")
        print(f"  {'Cohen\\'s d (effect size)':<35} "
              f"{arabic_results['cohens_d']:>12.4f} "
              f"{english_results['cohens_d']:>12.4f}")
        print(f"  {'p-value':<35} "
              f"{arabic_results['p_value']:>12.6f} "
              f"{english_results['p_value']:>12.6f}")
        print(f"\n  Arabic gap / English gap ratio: {gap_ratio:.2f}x")

        print(f"\n  INTERPRETATION:")
        if ar_gap > en_gap:
            print(f"  ✓ Arabic root-cluster gap ({ar_gap:.4f}) EXCEEDS")
            print(f"    English morphological gap ({en_gap:.4f})")
            print(f"    by a factor of {gap_ratio:.2f}x")
            print(f"")
            print(f"  This supports the framework's Claim 1: Arabic's")
            print(f"  triconsonantal root-pattern morphology produces")
            print(f"  stronger geometric clustering in embedding space")
            print(f"  than English's non-templatic morphology.")
        else:
            print(f"  ✗ Arabic gap does NOT exceed English gap.")
            print(f"  This would falsify Claim 1 at the surface level.")
            print(f"  Investigate: vocabulary coverage, model choice,")
            print(f"  word selection methodology.")


def generate_visualizations(arabic_results, english_results, output_dir):
    """Generate plots for the paper."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available — skipping visualizations")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Plot 1: Distribution comparison ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Root-Cluster Density: Arabic vs English\nAL-MIR'ĀH Framework — Q1 Proof of Concept",
                 fontsize=13, fontweight='bold')

    colors = {'intra': '#c8a96e', 'cross': '#4a6e8a'}

    for ax, results, title in [
        (axes[0], arabic_results, "Arabic (AraVec)"),
        (axes[1], english_results, "English (word2vec)")
    ]:
        if results is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        ax.hist(results['intra_sims_sample'], bins=40, alpha=0.7,
                color=colors['intra'], label=f"Intra-root\nμ={results['intra_mean']:.3f}",
                density=True)
        ax.hist(results['cross_sims_sample'], bins=40, alpha=0.7,
                color=colors['cross'], label=f"Cross-root\nμ={results['cross_mean']:.3f}",
                density=True)

        ax.axvline(results['intra_mean'], color=colors['intra'],
                   linestyle='--', linewidth=2)
        ax.axvline(results['cross_mean'], color=colors['cross'],
                   linestyle='--', linewidth=2)

        ax.set_title(f"{title}\nGap = {results['gap']:.4f}, d = {results['cohens_d']:.3f}",
                     fontsize=11)
        ax.set_xlabel("Cosine Similarity", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(-0.2, 1.0)

    plt.tight_layout()
    path1 = os.path.join(output_dir, "q1_similarity_distributions.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # ── Plot 2: Gap comparison bar chart ─────────────────────────────
    if arabic_results and english_results:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle("Root-Cluster Gap: Arabic vs English",
                     fontsize=13, fontweight='bold')

        langs = ['Arabic\n(AraVec)', 'English\n(word2vec)']
        gaps = [arabic_results['gap'], english_results['gap']]
        errors = [
            np.sqrt(arabic_results['intra_std']**2 + arabic_results['cross_std']**2),
            np.sqrt(english_results['intra_std']**2 + english_results['cross_std']**2),
        ]
        bar_colors = ['#c8a96e', '#4a6e8a']

        bars = ax.bar(langs, gaps, color=bar_colors, alpha=0.85,
                      width=0.4, yerr=errors, capsize=8,
                      error_kw={'linewidth': 2})

        for bar, gap in zip(bars, gaps):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{gap:.4f}', ha='center', va='bottom', fontsize=11,
                    fontweight='bold')

        ax.set_ylabel("Gap (Intra − Cross Cosine Similarity)", fontsize=11)
        ax.set_title("Larger gap = stronger root-cluster structure", fontsize=10)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylim(0, max(gaps) * 1.3)

        ratio = arabic_results['gap'] / english_results['gap'] if english_results['gap'] > 0 else 0
        ax.text(0.98, 0.95, f"Arabic/English ratio: {ratio:.2f}x",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        path2 = os.path.join(output_dir, "q1_gap_comparison.png")
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path2}")

    # ── Plot 3: Per-family heatmap ────────────────────────────────────
    for results, fname in [
        (arabic_results, "q1_arabic_family_heatmap.png"),
        (english_results, "q1_english_family_heatmap.png")
    ]:
        if results is None:
            continue
        stats = results['family_stats']
        roots = list(stats.keys())
        means = [stats[r]['intra_mean'] for r in roots]

        fig, ax = plt.subplots(figsize=(max(8, len(roots)*0.8), 4))
        bars = ax.bar(range(len(roots)), means, color='#c8a96e', alpha=0.8)
        ax.axhline(results['cross_mean'], color='#4a6e8a',
                   linestyle='--', linewidth=2, label=f"Cross-family mean ({results['cross_mean']:.3f})")
        ax.set_xticks(range(len(roots)))
        ax.set_xticklabels(roots, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Intra-family Mean Cosine Similarity")
        ax.set_title(f"Per-Root Cluster Density — {results['language']}")
        ax.legend()
        plt.tight_layout()
        path3 = os.path.join(output_dir, fname)
        plt.savefig(path3, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path3}")


def save_results(arabic_results, english_results, output_dir):
    """Save full results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output = {
        "experiment": "Q1 Measurement 1 — Root-Cluster Density",
        "framework": "AL-MIR'ĀH",
        "claim_tested": "Arabic intra-root cosine similarity > cross-root similarity; gap larger than English",
        "arabic": arabic_results,
        "english": english_results,
    }
    if arabic_results and english_results:
        output["comparative"] = {
            "arabic_gap": arabic_results["gap"],
            "english_gap": english_results["gap"],
            "gap_ratio": arabic_results["gap"] / english_results["gap"] if english_results["gap"] > 0 else None,
            "arabic_cohens_d": arabic_results["cohens_d"],
            "english_cohens_d": english_results["cohens_d"],
            "claim_supported": arabic_results["gap"] > english_results["gap"] and arabic_results["significant"],
        }

    path = os.path.join(output_dir, "q1_results.json")
    with open(path, 'w', ensure_ascii=False) as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")
    return output


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Q1 Root-Cluster Density: Arabic vs English word2vec"
    )
    parser.add_argument(
        '--arabic', type=str, required=True,
        help='Path to AraVec .bin file (binary=True word2vec format)'
    )
    parser.add_argument(
        '--english', type=str, required=True,
        help='Path to Google News word2vec .bin file OR "download" to fetch via gensim'
    )
    parser.add_argument(
        '--output', type=str, default='q1_results',
        help='Output directory for results and plots (default: q1_results)'
    )
    parser.add_argument(
        '--min-words', type=int, default=3,
        help='Minimum words per family required to include it (default: 3)'
    )
    args = parser.parse_args()

    try:
        from gensim.models import KeyedVectors
        import gensim.downloader as gensim_api
    except ImportError:
        print("ERROR: gensim not installed. Run: pip install gensim")
        sys.exit(1)

    # ── Load Arabic model ─────────────────────────────────────────────
    print(f"\nLoading Arabic model: {args.arabic}")
    try:
        arabic_model = KeyedVectors.load_word2vec_format(args.arabic, binary=True)
        print(f"  ✓ Loaded. Vocabulary size: {len(arabic_model.key_to_index):,}")
    except Exception as e:
        print(f"  ✗ Failed to load Arabic model: {e}")
        print(f"  Try: KeyedVectors.load_word2vec_format(path, binary=False) for text format")
        sys.exit(1)

    # ── Load English model ────────────────────────────────────────────
    print(f"\nLoading English model: {args.english}")
    try:
        if args.english == "download":
            print("  Downloading word2vec-google-news-300 via gensim...")
            english_model = gensim_api.load('word2vec-google-news-300')
        else:
            english_model = KeyedVectors.load_word2vec_format(args.english, binary=True)
        print(f"  ✓ Loaded. Vocabulary size: {len(english_model.key_to_index):,}")
    except Exception as e:
        print(f"  ✗ Failed to load English model: {e}")
        sys.exit(1)

    # ── Run Measurement 1 ─────────────────────────────────────────────
    arabic_results = compute_root_cluster_density(
        arabic_model, ARABIC_ROOT_FAMILIES, "Arabic (AraVec)", args.min_words
    )
    english_results = compute_root_cluster_density(
        english_model, ENGLISH_MORPH_FAMILIES, "English (word2vec)", args.min_words
    )

    # ── Comparative summary ───────────────────────────────────────────
    print_comparative_summary(arabic_results, english_results)

    # ── Visualizations ────────────────────────────────────────────────
    print(f"\nGenerating visualizations...")
    generate_visualizations(arabic_results, english_results, args.output)

    # ── Save results ──────────────────────────────────────────────────
    print(f"\nSaving results...")
    results = save_results(arabic_results, english_results, args.output)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"Results in: {args.output}/")
    print(f"  q1_results.json")
    print(f"  q1_similarity_distributions.png")
    print(f"  q1_gap_comparison.png")
    print(f"  q1_arabic_family_heatmap.png")
    print(f"  q1_english_family_heatmap.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
