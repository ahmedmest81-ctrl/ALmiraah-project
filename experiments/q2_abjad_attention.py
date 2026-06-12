"""
AL-MIR'ĀH — Q2: Abjad Proximity and Attention Weights
=======================================================
Research Question Q2: Do transformer attention weights in CAMeLBERT-ca
correlate with Abjad proximity in classical Arabic texts, after controlling
for positional distance and co-occurrence frequency?

Prediction: positive partial correlation in classical corpus (Fuṣūṣ + Futūḥāt);
null in Modern Standard Arabic control corpus.

Designed to run on HuggingFace Space with GPU (T4 or A10).
Corpus loaded from HuggingFace Dataset: WELLyes1/almiraah_coordinate_db

Author: AL-MIR'ĀH Framework — Ahmed & Al-Mirʾāh — March 2026
"""

import os
import re
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from q2_robustness import generate_verdict_corrected as generate_verdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("q2")

# ── Constants ─────────────────────────────────────────────────────────────────
CAMELBERT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
RANDOM_SEED = 42
MAX_SEQ_LEN = 512          # BERT max; we'll use 128 for speed
WORKING_SEQ_LEN = 128
BATCH_SIZE = 8              # Conservative for the RTX 4060's 8GB VRAM
N_SENTENCES = 120000         # Sentences to process for attention extraction
MIN_SENTENCE_TOKENS = 6    # Skip very short sentences
MAX_SENTENCE_TOKENS = 80   # Skip very long sentences (truncation artifacts)

# Abjad distance thresholds (pre-specified, from Q1/PMI design)
PROXIMATE_THRESHOLD = 10   # Δ ≤ 10
DISTAL_THRESHOLD = 50      # Δ > 50

# Mashriqi Abjad order
ABJAD_VALUES = {
    'ا': 1,  'ب': 2,  'ج': 3,  'د': 4,  'ه': 5,  'و': 6,  'ز': 7,
    'ح': 8,  'ط': 9,  'ي': 10, 'ك': 20, 'ل': 30, 'م': 40, 'ن': 50,
    'س': 60, 'ع': 70, 'ف': 80, 'ص': 90, 'ق': 100,'ر': 200,'ش': 300,
    'ت': 400,'ث': 500,'خ': 600,'ذ': 700,'ض': 800,'ظ': 900,'غ': 1000,
    # Variants
    'أ': 1, 'إ': 1, 'آ': 1, 'ة': 400, 'ى': 10, 'ؤ': 6, 'ئ': 10,
}

# ── Abjad computation ─────────────────────────────────────────────────────────

def abjad_value(word: str) -> int:
    """Compute Mashriqi Abjad value for a word (diacritics already stripped)."""
    return sum(ABJAD_VALUES.get(ch, 0) for ch in word)


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks (harakat, shadda, etc.)."""
    return re.sub(r'[\u064B-\u065F\u0670]', '', text)


def clean_arabic(text: str) -> str:
    """Strip diacritics and non-Arabic characters, collapse whitespace."""
    text = strip_diacritics(text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Root extraction (same heuristic as PMI pipeline) ─────────────────────────

# High-frequency curated roots from the PMI pipeline
CURATED_ROOTS = {
    # Format: surface_form -> root_consonants
    'الله': 'اله', 'وجود': 'وجد', 'عالم': 'علم', 'حق': 'حقق',
    'واحد': 'وحد', 'كل': 'كلل', 'شيء': 'شيء', 'اسم': 'سمو',
    'صفة': 'وصف', 'ذات': 'ذات', 'فعل': 'فعل', 'قول': 'قول',
    'علم': 'علم', 'ارادة': 'ارد', 'حياة': 'حيي', 'قدرة': 'قدر',
    'سمع': 'سمع', 'بصر': 'بصر', 'كلام': 'كلم', 'ايجاد': 'وجد',
    'تجلي': 'جلو', 'بيان': 'بين', 'نور': 'نور', 'روح': 'روح',
    'قلب': 'قلب', 'عقل': 'عقل', 'نفس': 'نفس', 'جسم': 'جسم',
    'حكمة': 'حكم', 'معرفة': 'عرف', 'محبة': 'حبب', 'فناء': 'فني',
    'بقاء': 'بقي', 'برزخ': 'برزخ', 'حضرة': 'حضر', 'مقام': 'قوم',
    'حال': 'حول', 'ذكر': 'ذكر', 'فكر': 'فكر', 'شهود': 'شهد',
    'كشف': 'كشف', 'وحي': 'وحي', 'نبي': 'نبا', 'رسول': 'رسل',
    'امر': 'امر', 'خلق': 'خلق', 'رحمة': 'رحم', 'نعمة': 'نعم',
    'ايمان': 'امن', 'اسلام': 'سلم', 'عبادة': 'عبد', 'طاعة': 'طوع',
    'توبة': 'توب', 'صبر': 'صبر', 'شكر': 'شكر', 'تواضع': 'وضع',
    'محمد': 'حمد', 'ابراهيم': 'برهم', 'موسى': 'موس', 'عيسى': 'عيس',
    'ادم': 'ادم', 'اول': 'اول', 'اخر': 'اخر', 'ظاهر': 'ظهر',
    'باطن': 'بطن', 'كتاب': 'كتب', 'قران': 'قرن', 'اية': 'ايي',
    'صلاة': 'صلو', 'زكاة': 'زكو', 'صوم': 'صوم', 'حج': 'حجج',
}

ARABIC_PREFIXES = ['وال', 'فال', 'بال', 'لل', 'ال', 'و', 'ف', 'ب', 'ل', 'ك']

def extract_root_heuristic(word: str) -> str:
    """
    Two-stage root extraction:
    1. Curated table lookup
    2. Prefix-stripping + 3-consonant skeleton
    Returns empty string if extraction is unreliable.
    """
    w = clean_arabic(word).strip()
    if not w:
        return ''

    # Stage 1: curated table
    if w in CURATED_ROOTS:
        return CURATED_ROOTS[w]

    # Stage 2: prefix stripping
    for prefix in ARABIC_PREFIXES:
        if w.startswith(prefix) and len(w) > len(prefix) + 2:
            w = w[len(prefix):]
            break

    # Consonant skeleton: remove long vowels (ا و ي in medial/final position)
    # Heuristic: keep first 3 distinct consonants that are not 'pure' vowel carriers
    # This is approximate for the long tail
    WEAK = {'ا', 'و', 'ي', 'ى', 'ة', 'ء', 'أ', 'إ', 'آ', 'ؤ', 'ئ'}
    consonants = [ch for ch in w if ch not in WEAK and ch in ABJAD_VALUES]

    if len(consonants) < 3:
        return ''  # Cannot reliably determine root

    # Return first 3 consonants as root approximation
    return ''.join(consonants[:3])


def same_root(w1: str, w2: str) -> bool:
    """Returns True if both words appear to share a root. Unreliable pairs return None."""
    r1 = extract_root_heuristic(w1)
    r2 = extract_root_heuristic(w2)
    if not r1 or not r2:
        return None  # Unreliable — exclude from cross-root analysis
    return r1 == r2

# ── Corpus loading ─────────────────────────────────────────────────────────────

def load_corpus_from_dataset(dataset_name: str = "WELLyes1/almiraah_coordinate_db",
                              hf_token: str = None) -> list[str]:
    """
    Load the Ibn ʿArabī corpus from the HuggingFace dataset.
    Falls back to local path if dataset doesn't contain the text corpus.
    Returns a list of cleaned Arabic sentences.
    """
    log.info(f"Loading corpus from HuggingFace dataset: {dataset_name}")

    try:
        from huggingface_hub import hf_hub_download
        from datasets import load_dataset

        # Try loading as dataset first
        ds = load_dataset(dataset_name, token=hf_token)
        log.info(f"Dataset loaded: {ds}")

        # Extract text field if present
        sentences = []
        for split in ds:
            for row in ds[split]:
                for field in ['text', 'content', 'sentence', 'passage']:
                    if field in row and row[field]:
                        text = clean_arabic(str(row[field]))
                        if len(text.split()) >= MIN_SENTENCE_TOKENS:
                            sentences.append(text)
                        break

        if sentences:
            log.info(f"Loaded {len(sentences):,} sentences from dataset")
            return sentences

    except Exception as e:
        log.warning(f"Dataset load failed: {e}")

    # Fallback: look for corpus file in Space filesystem
    corpus_paths = [
        "/data/ibn_arabi_corpus.txt",
        "/data/fusus_futuhat_cleaned.txt",
        "fusus_futuhat_cleaned.txt",
        "corpus.txt"
    ]

    for path in corpus_paths:
        if Path(path).exists():
            log.info(f"Loading corpus from {path}")
            return load_corpus_from_file(path)

    log.error("No corpus found. Please upload fusus_futuhat_cleaned.txt to the Space.")
    raise FileNotFoundError(
        "Corpus not found. Expected either:\n"
        "  - HuggingFace dataset with text field\n"
        "  - /data/fusus_futuhat_cleaned.txt in Space filesystem\n"
        "  - corpus.txt in working directory\n\n"
        "Prepare corpus with: python q2_abjad_attention.py --prepare-corpus"
    )


def load_corpus_from_file(path: str) -> list[str]:
    """Load and sentence-segment a cleaned Arabic text file."""
    with open(path, encoding='utf-8') as f:
        text = f.read()

    # Split on sentence boundaries (Arabic full stop + newlines)
    raw_sentences = re.split(r'[.\n،؛]+', text)
    sentences = []
    for s in raw_sentences:
        s = clean_arabic(s)
        tokens = s.split()
        if MIN_SENTENCE_TOKENS <= len(tokens) <= MAX_SENTENCE_TOKENS:
            sentences.append(s)

    log.info(f"Loaded {len(sentences):,} sentences from {path}")
    return sentences


def prepare_corpus_from_openiti(output_path: str = "fusus_futuhat_cleaned.txt"):
    """
    Download and clean Ibn ʿArabī texts from OpenITI.
    Run once with: python q2_abjad_attention.py --prepare-corpus
    """
    import urllib.request

    OPENITI_URLS = {
        'fusus': 'https://raw.githubusercontent.com/OpenITI/0625AH/master/data/0638IbnArabiAndalusi/0638IbnArabiAndalusi.FususHikam/0638IbnArabiAndalusi.FususHikam.JK000416-ara1',
        'futuhat': 'https://raw.githubusercontent.com/OpenITI/0625AH/master/data/0638IbnArabiAndalusi/0638IbnArabiAndalusi.FutuhatMakkiyya/0638IbnArabiAndalusi.FutuhatMakkiyya.JK000649-ara1',
    }

    combined = []
    for name, url in OPENITI_URLS.items():
        log.info(f"Downloading {name} from OpenITI...")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read().decode('utf-8')
        except Exception as e:
            log.error(f"Failed to download {name}: {e}")
            log.info("Please download OpenITI files manually and place at the path above.")
            continue

        # Remove OpenITI mARkdown markup
        raw = re.sub(r'^#.*$', '', raw, flags=re.MULTILINE)   # metadata lines
        raw = re.sub(r'~+', ' ', raw)                          # continuation markers
        raw = re.sub(r'ms\d+', '', raw)                        # manuscript markers
        raw = re.sub(r'PageV\d+P\d+', '', raw)                 # page markers
        raw = clean_arabic(raw)
        combined.append(raw)
        log.info(f"  {name}: {len(raw.split()):,} tokens after cleaning")

    full_text = '\n'.join(combined)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    log.info(f"Corpus written to {output_path} ({len(full_text.split()):,} tokens)")

# ── Attention extraction ───────────────────────────────────────────────────────

class AttentionExtractor:
    """
    Loads CAMeLBERT-ca and extracts attention weights for token pairs.
    """

    def __init__(self, model_name: str = CAMELBERT_MODEL, device: str = None):
        import torch
        from transformers import AutoTokenizer, AutoModel

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        log.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_attentions=True,
            attn_implementation="eager"  # Required for attention output
        ).to(self.device)
        self.model.eval()
        log.info("Model loaded.")

    def extract(self, sentences: list[str]) -> list[dict]:
        """
        For each sentence, return a list of token-pair records:
        {
            'word_i': str, 'word_j': str,
            'pos_i': int, 'pos_j': int,
            'pos_distance': int,
            'abjad_i': int, 'abjad_j': int,
            'abjad_distance': int,
            'mean_attention': float,   # mean over all layers × heads, both directions
            'root_relation': str,      # 'same', 'different', 'unknown'
        }
        """
        import torch

        all_records = []
        total = min(len(sentences), N_SENTENCES)

        for idx, sentence in enumerate(sentences[:total]):
            if idx % 500 == 0:
                log.info(f"  Sentence {idx}/{total}...")

            tokens_raw = sentence.split()
            if not (MIN_SENTENCE_TOKENS <= len(tokens_raw) <= MAX_SENTENCE_TOKENS):
                continue

            # Tokenize
            encoded = self.tokenizer(
                sentence,
                return_tensors='pt',
                max_length=WORKING_SEQ_LEN,
                truncation=True,
                padding=False
            ).to(self.device)

            n_input_tokens = encoded['input_ids'].shape[1]
            if n_input_tokens < 4:
                continue

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**encoded)

            # outputs.attentions: tuple of (n_layers,) each shape (1, n_heads, seq, seq)
            # Stack to (n_layers, n_heads, seq, seq)
            attn = torch.stack(outputs.attentions, dim=0).squeeze(1)
            # attn shape: (n_layers, n_heads, seq_len, seq_len)

            # Mean over layers and heads → (seq_len, seq_len)
            mean_attn = attn.mean(dim=(0, 1)).cpu().numpy()

            # Map input_ids back to word strings
            # Use tokenizer to get word-level tokens (subword aware)
            word_ids = encoded.word_ids()  # list of word indices per subword token
            # Get unique word positions (skip None = special tokens)
            word_positions = {}
            for tok_idx, word_idx in enumerate(word_ids):
                if word_idx is not None and word_idx not in word_positions:
                    word_positions[word_idx] = tok_idx  # first subword token position

            # Build word list from original tokens (already cleaned Arabic)
            words = tokens_raw
            n_words = len(words)

            # For each word pair (i, j), compute mean attention
            # We use the first subword token position as the word's attention position
            records_this = []
            for wi in range(n_words):
                for wj in range(wi + 1, n_words):
                    if wi not in word_positions or wj not in word_positions:
                        continue

                    ti = word_positions[wi]
                    tj = word_positions[wj]

                    if ti >= mean_attn.shape[0] or tj >= mean_attn.shape[0]:
                        continue

                    # Bidirectional mean attention (i→j and j→i)
                    attn_val = float((mean_attn[ti, tj] + mean_attn[tj, ti]) / 2)

                    word_i = words[wi]
                    word_j = words[wj]

                    abjad_i = abjad_value(word_i)
                    abjad_j = abjad_value(word_j)
                    abjad_dist = abs(abjad_i - abjad_j)

                    pos_dist = wj - wi

                    # Root relationship
                    sr = same_root(word_i, word_j)
                    if sr is None:
                        root_rel = 'unknown'
                    elif sr:
                        root_rel = 'same'
                    else:
                        root_rel = 'different'

                    records_this.append({
                        'word_i': word_i,
                        'word_j': word_j,
                        'pos_i': wi,
                        'pos_j': wj,
                        'pos_distance': pos_dist,
                        'abjad_i': abjad_i,
                        'abjad_j': abjad_j,
                        'abjad_distance': abjad_dist,
                        'mean_attention': attn_val,
                        'root_relation': root_rel,
                    })

            all_records.extend(records_this)

        log.info(f"Extracted {len(all_records):,} word-pair attention records")
        return all_records

# ── Statistical analysis ───────────────────────────────────────────────────────

def compute_cooccurrence_frequency(sentences: list[str], vocab: set) -> dict:
    """
    Compute passage-level co-occurrence frequency for all word pairs in vocab.
    Uses same 40-word window as PMI pipeline.
    """
    log.info("Computing co-occurrence frequencies...")
    WINDOW = 40
    STEP = 20

    # Build passage list
    all_tokens = ' '.join(sentences).split()
    passages = []
    for start in range(0, len(all_tokens) - WINDOW, STEP):
        passage_words = set(all_tokens[start:start + WINDOW]) & vocab
        passages.append(passage_words)

    log.info(f"  {len(passages):,} passages")

    # Word frequencies
    word_freq = Counter(w for p in passages for w in p)
    n_passages = len(passages)

    # Co-occurrence counts
    cooc = defaultdict(int)
    for passage in passages:
        words_list = sorted(passage)
        for i, w1 in enumerate(words_list):
            for w2 in words_list[i + 1:]:
                cooc[(w1, w2)] += 1

    # Convert to PMI-style frequency ratios (normalized)
    freq_map = {}
    for (w1, w2), count in cooc.items():
        if count >= 2:
            p_w1 = word_freq[w1] / n_passages
            p_w2 = word_freq[w2] / n_passages
            p_joint = count / n_passages
            freq_map[(w1, w2)] = p_joint / (p_w1 * p_w2 + 1e-10)
            freq_map[(w2, w1)] = freq_map[(w1, w2)]

    log.info(f"  {len(freq_map):,} co-occurrence pairs computed")
    return freq_map


def run_partial_correlation_analysis(records: list[dict],
                                      cooc_freq: dict) -> dict:
    """
    Core statistical analysis.

    For cross-root pairs only:
    - Partial correlation: Abjad distance ~ attention weight
      controlling for positional distance and co-occurrence frequency

    Returns results dict for reporting.
    """
    log.info("Running partial correlation analysis...")

    # Filter to cross-root pairs only (the clean test)
    cross_root = [r for r in records if r['root_relation'] == 'different']
    same_root_rec = [r for r in records if r['root_relation'] == 'same']
    unknown = [r for r in records if r['root_relation'] == 'unknown']

    log.info(f"  Cross-root pairs: {len(cross_root):,}")
    log.info(f"  Same-root pairs:  {len(same_root_rec):,}")
    log.info(f"  Unknown root:     {len(unknown):,}")

    def _partial_corr_spearman(data: list[dict], control_vars: list[str],
                                target: str = 'abjad_distance',
                                outcome: str = 'mean_attention') -> dict:
        """
        Spearman partial correlation between target and outcome,
        controlling for control_vars, via residualization.
        """
        if len(data) < 50:
            return {'rho': None, 'p': None, 'n': len(data), 'error': 'insufficient data'}

        from scipy.stats import spearmanr
        from sklearn.linear_model import LinearRegression

        X_target = np.array([r[target] for r in data]).reshape(-1, 1)
        X_outcome = np.array([r[outcome] for r in data]).reshape(-1, 1)

        # Controls
        control_matrix = []
        for var in control_vars:
            if var == 'cooc_freq':
                vals = []
                for r in data:
                    key = (r['word_i'], r['word_j'])
                    vals.append(cooc_freq.get(key, cooc_freq.get((r['word_j'], r['word_i']), 0.0)))
                control_matrix.append(vals)
            else:
                control_matrix.append([r[var] for r in data])

        X_controls = np.array(control_matrix).T  # (n, n_controls)

        # Residualize target on controls
        reg = LinearRegression().fit(X_controls, X_target.ravel())
        resid_target = X_target.ravel() - reg.predict(X_controls)

        # Residualize outcome on controls
        reg2 = LinearRegression().fit(X_controls, X_outcome.ravel())
        resid_outcome = X_outcome.ravel() - reg2.predict(X_controls)

        rho, p = spearmanr(resid_target, resid_outcome)
        return {'rho': float(rho), 'p': float(p), 'n': len(data)}

    results = {}

    # ── Primary result: cross-root, Abjad distance ~ attention ──────────────
    results['primary'] = _partial_corr_spearman(
        cross_root,
        control_vars=['pos_distance', 'cooc_freq'],
        target='abjad_distance',
        outcome='mean_attention'
    )
    log.info(f"  Primary result (cross-root): ρ={results['primary'].get('rho'):.4f}, "
             f"p={results['primary'].get('p'):.4e}, n={results['primary']['n']:,}")

    # ── Same-root control: should show positive attention signal ─────────────
    results['same_root_control'] = _partial_corr_spearman(
        same_root_rec,
        control_vars=['pos_distance', 'cooc_freq'],
        target='abjad_distance',
        outcome='mean_attention'
    )
    log.info(f"  Same-root control: ρ={results['same_root_control'].get('rho'):.4f}")

    # ── Binned analysis: mean attention by Abjad distance bin ───────────────
    bins = [
        ('proximate_0_10',    [r for r in cross_root if r['abjad_distance'] <= 10]),
        ('medial_11_25',      [r for r in cross_root if 11 <= r['abjad_distance'] <= 25]),
        ('medial_26_50',      [r for r in cross_root if 26 <= r['abjad_distance'] <= 50]),
        ('distal_51_100',     [r for r in cross_root if 51 <= r['abjad_distance'] <= 100]),
        ('distal_101_200',    [r for r in cross_root if 101 <= r['abjad_distance'] <= 200]),
        ('distal_201_plus',   [r for r in cross_root if r['abjad_distance'] > 200]),
    ]

    results['binned'] = {}
    for label, bin_records in bins:
        if bin_records:
            attns = [r['mean_attention'] for r in bin_records]
            results['binned'][label] = {
                'n': len(bin_records),
                'mean_attention': float(np.mean(attns)),
                'median_attention': float(np.median(attns)),
                'sd': float(np.std(attns)),
            }
        else:
            results['binned'][label] = {'n': 0}

    # ── Unadjusted Spearman (for comparison) ────────────────────────────────
    if cross_root:
        abjad_dists = [r['abjad_distance'] for r in cross_root]
        attns = [r['mean_attention'] for r in cross_root]
        rho_raw, p_raw = spearmanr(abjad_dists, attns)
        results['unadjusted'] = {'rho': float(rho_raw), 'p': float(p_raw), 'n': len(cross_root)}
        log.info(f"  Unadjusted Spearman: ρ={rho_raw:.4f}, p={p_raw:.4e}")

    return results

# ── Output ─────────────────────────────────────────────────────────────────────

def save_results(results: dict, verdict: str, output_dir: str = "q2_results"):
    """Save all results to JSON and a plain-text report."""
    Path(output_dir).mkdir(exist_ok=True)

    # Full results JSON
    out_path = f"{output_dir}/q2_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'verdict': verdict}, f, ensure_ascii=False, indent=2)
    log.info(f"Results written to {out_path}")

    # Plain-text report
    report_path = f"{output_dir}/q2_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("AL-MIR'ĀH — Q2: Abjad Proximity and Attention Weights\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {CAMELBERT_MODEL}\n")
        f.write(f"Corpus: Ibn ʿArabī Fuṣūṣ + Futūḥāt (OpenITI)\n")
        f.write(f"Sentences sampled: {N_SENTENCES}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")

        f.write("PRIMARY RESULT (cross-root pairs)\n")
        f.write("-" * 40 + "\n")
        p = results.get('primary', {})
        f.write(f"Partial ρ (Abjad dist ~ attention, controlling pos_dist + cooc_freq):\n")
        f.write(f"  ρ = {p.get('rho', 'N/A')}\n")
        f.write(f"  p = {p.get('p', 'N/A')}\n")
        f.write(f"  n = {p.get('n', 'N/A'):,}\n\n")

        f.write("BINNED ANALYSIS (mean attention by Abjad distance)\n")
        f.write("-" * 40 + "\n")
        for bin_label, bin_stats in results.get('binned', {}).items():
            if bin_stats.get('n', 0) > 0:
                f.write(f"  {bin_label}: n={bin_stats['n']:,}, "
                        f"mean_attn={bin_stats['mean_attention']:.6f}, "
                        f"median={bin_stats['median_attention']:.6f}\n")

        f.write("\nSAME-ROOT CONTROL\n")
        f.write("-" * 40 + "\n")
        sr = results.get('same_root_control', {})
        f.write(f"  ρ = {sr.get('rho', 'N/A')}, p = {sr.get('p', 'N/A')}, n = {sr.get('n', 0):,}\n\n")

        f.write("VERDICT\n")
        f.write("-" * 40 + "\n")
        f.write(verdict + "\n")

    log.info(f"Report written to {report_path}")
    print("\n" + "=" * 60)
    print(verdict)
    print("=" * 60)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Q2: Abjad Proximity × Attention Analysis")
    parser.add_argument('--corpus', type=str, default=None,
                        help='Path to cleaned Arabic corpus file (one sentence per line or free text)')
    parser.add_argument('--dataset', type=str, default='WELLyes1/almiraah_coordinate_db',
                        help='HuggingFace dataset name to load corpus from')
    parser.add_argument('--hf-token', type=str, default=os.environ.get('HF_TOKEN'),
                        help='HuggingFace API token')
    parser.add_argument('--n-sentences', type=int, default=N_SENTENCES,
                        help=f'Number of sentences to process (default: {N_SENTENCES})')
    parser.add_argument('--output', type=str, default='q2_results',
                        help='Output directory for results')
    parser.add_argument('--prepare-corpus', action='store_true',
                        help='Download and prepare corpus from OpenITI, then exit')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda, cpu (default: auto-detect)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Corpus preparation mode
    if args.prepare_corpus:
        prepare_corpus_from_openiti()
        return

    # ── Load corpus ──────────────────────────────────────────────────────────
    if args.corpus:
        sentences = load_corpus_from_file(args.corpus)
    else:
        sentences = load_corpus_from_dataset(args.dataset, args.hf_token)

    # Shuffle and subsample
    random.shuffle(sentences)
    sentences = sentences[:args.n_sentences]
    log.info(f"Using {len(sentences):,} sentences")

    # ── Build vocabulary for co-occurrence ───────────────────────────────────
    vocab = set()
    for s in sentences:
        vocab.update(s.split())
    log.info(f"Vocabulary: {len(vocab):,} types")

    # ── Co-occurrence frequencies ────────────────────────────────────────────
    cooc_freq = compute_cooccurrence_frequency(sentences, vocab)

    # Persist for robustness checks
    import pickle
    Path(args.output).mkdir(exist_ok=True)
    with open(Path(args.output) / 'q2_cooc.pkl', 'wb') as f:
        pickle.dump(cooc_freq, f)
    log.info(f"Saved co-occurrence dict to {args.output}/q2_cooc.pkl")

    # ── Attention extraction ─────────────────────────────────────────────────
    extractor = AttentionExtractor(model_name=CAMELBERT_MODEL, device=args.device)
    records = extractor.extract(sentences)

    if not records:
        log.error("No records extracted. Check corpus format.")
        return

    # Persist records for robustness checks
    with open(Path(args.output) / 'q2_records.pkl', 'wb') as f:
        pickle.dump(records, f)
    log.info(f"Saved records to {args.output}/q2_records.pkl ({len(records):,} pairs)")
    # ── Statistical analysis ─────────────────────────────────────────────────
    results = run_partial_correlation_analysis(records, cooc_freq)

    # ── Verdict ──────────────────────────────────────────────────────────────
    verdict = generate_verdict(results)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_results(results, verdict, args.output)


if __name__ == '__main__':
    main()
