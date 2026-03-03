# ALmiraah-project

**AL-MIR'ĀH Research Framework** — Q1 Proof of Concept: Root-Cluster Density

Tests whether Arabic words sharing a triconsonantal root cluster more tightly in
embedding space than English words sharing a prefix/suffix.

---

## Setup

**Step 1 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 2 — Download word embedding models:**

*Arabic* — [AraVec](https://github.com/bakrianoo/aravec) (trained on 3.3B tokens):
- Download `full_grams_cbow_300.zip` from the AraVec releases page
- Extract the `.bin` file

*English* — fetched automatically by gensim on first run (`word2vec-google-news-300`, ~1.6 GB)

---

## Run

```bash
python q1_root_cluster_density.py \
  --arabic /path/to/full_grams_cbow_300.bin \
  --english download \
  --output q1_results
```

| Argument | Description |
|---|---|
| `--arabic` | Path to AraVec `.bin` file |
| `--english` | Path to Google News `.bin` file, or `download` to fetch via gensim |
| `--output` | Output directory for results and plots (default: `q1_results`) |
| `--min-words` | Minimum words per family required (default: 3) |

---

## Output

Results are written to `q1_results/`:

| File | Contents |
|---|---|
| `q1_results.json` | Full statistics (intra/cross similarities, Cohen's d, p-values) |
| `q1_similarity_distributions.png` | Histogram of intra- vs cross-root similarities |
| `q1_gap_comparison.png` | Bar chart comparing Arabic vs English gap |
| `q1_arabic_family_heatmap.png` | Per-root cluster density (Arabic) |
| `q1_english_family_heatmap.png` | Per-root cluster density (English) |
