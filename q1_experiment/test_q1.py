"""
Unit tests for q1_root_cluster_density.py
Run with: pytest test_q1.py -v
"""

import json
import os
import tempfile
import numpy as np
import pytest

from q1_root_cluster_density import (
    cosine_similarity,
    get_available_words,
    compute_pairwise_similarities,
    compute_root_cluster_density,
    save_results,
    ARABIC_ROOT_FAMILIES,
    ENGLISH_MORPH_FAMILIES,
)


# ── Minimal mock KeyedVectors ──────────────────────────────────────────────────

class MockModel:
    """Lightweight stand-in for gensim KeyedVectors."""

    def __init__(self, vocab: dict):
        # vocab maps word -> numpy vector
        self._vocab = {w: np.array(v, dtype=np.float32) for w, v in vocab.items()}
        self.key_to_index = {w: i for i, w in enumerate(self._vocab)}

    def __getitem__(self, word):
        if word not in self._vocab:
            raise KeyError(word)
        return self._vocab[word]


def _unit(seed, dim=10):
    """Random unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


# ── cosine_similarity ──────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        v = np.array([1.0, 0.0])
        z = np.array([0.0, 0.0])
        assert cosine_similarity(v, z) == 0.0
        assert cosine_similarity(z, z) == 0.0

    def test_symmetry(self):
        v1, v2 = _unit(0), _unit(1)
        assert cosine_similarity(v1, v2) == pytest.approx(cosine_similarity(v2, v1))

    def test_value_in_range(self):
        for i in range(20):
            v1, v2 = _unit(i), _unit(i + 100)
            sim = cosine_similarity(v1, v2)
            assert -1.0 <= sim <= 1.0


# ── get_available_words ────────────────────────────────────────────────────────

class TestGetAvailableWords:
    def setup_method(self):
        self.model = MockModel({"cat": _unit(0), "dog": _unit(1), "fish": _unit(2)})

    def test_all_present(self):
        assert get_available_words(self.model, ["cat", "dog"]) == ["cat", "dog"]

    def test_partial_match(self):
        result = get_available_words(self.model, ["cat", "bird", "dog"])
        assert result == ["cat", "dog"]

    def test_none_present(self):
        assert get_available_words(self.model, ["bird", "lion"]) == []

    def test_empty_list(self):
        assert get_available_words(self.model, []) == []

    def test_preserves_order(self):
        words = ["fish", "cat", "dog"]
        result = get_available_words(self.model, words)
        assert result == ["fish", "cat", "dog"]


# ── compute_pairwise_similarities ─────────────────────────────────────────────

class TestComputePairwiseSimilarities:
    def setup_method(self):
        # Three mutually orthogonal unit vectors → all pairs have sim ≈ 0
        self.model = MockModel({
            "a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "b": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "c": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        })

    def test_pair_count(self):
        sims = compute_pairwise_similarities(self.model, ["a", "b", "c"])
        # C(3,2) = 3 pairs
        assert len(sims) == 3

    def test_orthogonal_vectors_give_zero(self):
        sims = compute_pairwise_similarities(self.model, ["a", "b", "c"])
        for s in sims:
            assert s == pytest.approx(0.0, abs=1e-6)

    def test_identical_words_give_one(self):
        model = MockModel({"x": _unit(5), "y": _unit(5)})  # same seed → same vector
        sims = compute_pairwise_similarities(model, ["x", "y"])
        assert sims[0] == pytest.approx(1.0)

    def test_two_words_returns_one_pair(self):
        sims = compute_pairwise_similarities(self.model, ["a", "b"])
        assert len(sims) == 1


# ── compute_root_cluster_density ──────────────────────────────────────────────

def _build_cluster_model():
    """
    Two tight clusters (A-words, B-words) that are far from each other.
    A-words: small noise around e1 = [1,0,0,...]
    B-words: small noise around e2 = [0,1,0,...]
    """
    rng = np.random.default_rng(99)
    dim = 20
    e1 = np.zeros(dim); e1[0] = 1.0
    e2 = np.zeros(dim); e2[1] = 1.0

    vocab = {}
    for i in range(5):
        v = e1 + rng.standard_normal(dim) * 0.05
        vocab[f"a{i}"] = v / np.linalg.norm(v)
    for i in range(5):
        v = e2 + rng.standard_normal(dim) * 0.05
        vocab[f"b{i}"] = v / np.linalg.norm(v)

    return MockModel(vocab)


class TestComputeRootClusterDensity:
    def setup_method(self):
        self.model = _build_cluster_model()
        self.families = {
            "A": [f"a{i}" for i in range(5)],
            "B": [f"b{i}" for i in range(5)],
        }

    def test_returns_dict(self):
        result = compute_root_cluster_density(self.model, self.families, "Test", min_words=3)
        assert result is not None
        assert isinstance(result, dict)

    def test_intra_exceeds_cross(self):
        result = compute_root_cluster_density(self.model, self.families, "Test", min_words=3)
        assert result["intra_mean"] > result["cross_mean"]

    def test_gap_positive(self):
        result = compute_root_cluster_density(self.model, self.families, "Test", min_words=3)
        assert result["gap"] > 0

    def test_expected_family_count(self):
        result = compute_root_cluster_density(self.model, self.families, "Test", min_words=3)
        assert result["n_families"] == 2

    def test_min_words_filter(self):
        # Require 6 words per family — none qualify (only 5 each)
        result = compute_root_cluster_density(self.model, self.families, "Test", min_words=6)
        assert result is None

    def test_missing_vocab_skipped(self):
        # Only family A qualifies; MISSING words aren't in vocab.
        # With only 1 qualifying family, cross-family comparison is impossible → None.
        families = {"A": [f"a{i}" for i in range(5)], "MISSING": ["x", "y", "z"]}
        result = compute_root_cluster_density(self.model, families, "Test", min_words=3)
        assert result is None

    def test_result_keys_present(self):
        result = compute_root_cluster_density(self.model, self.families, "Test", min_words=3)
        expected = {"language", "n_families", "n_intra_pairs", "n_cross_pairs",
                    "intra_mean", "intra_std", "cross_mean", "cross_std",
                    "gap", "cohens_d", "p_value", "significant", "family_stats",
                    "intra_sims_sample", "cross_sims_sample"}
        assert expected.issubset(result.keys())

    def test_significance_flag_type(self):
        result = compute_root_cluster_density(self.model, self.families, "Test", min_words=3)
        assert isinstance(result["significant"], bool)


# ── save_results ──────────────────────────────────────────────────────────────

def _dummy_result(lang):
    return {
        "language": lang,
        "n_families": 2,
        "n_intra_pairs": 10,
        "n_cross_pairs": 20,
        "intra_mean": 0.8,
        "intra_std": 0.05,
        "cross_mean": 0.3,
        "cross_std": 0.1,
        "gap": 0.5,
        "cohens_d": 2.0,
        "p_value": 0.001,
        "significant": True,
        "family_stats": {},
        "intra_sims_sample": [0.8, 0.82],
        "cross_sims_sample": [0.3, 0.28],
    }


class TestSaveResults:
    def test_json_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_results(_dummy_result("Arabic"), _dummy_result("English"), tmp)
            assert os.path.exists(os.path.join(tmp, "q1_results.json"))

    def test_json_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_results(_dummy_result("Arabic"), _dummy_result("English"), tmp)
            with open(os.path.join(tmp, "q1_results.json"), encoding='utf-8') as f:
                data = json.load(f)
            assert "arabic" in data
            assert "english" in data

    def test_comparative_block_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_results(_dummy_result("Arabic"), _dummy_result("English"), tmp)
            with open(os.path.join(tmp, "q1_results.json"), encoding='utf-8') as f:
                data = json.load(f)
            assert "comparative" in data
            assert "gap_ratio" in data["comparative"]

    def test_none_arabic_saves_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_results(None, _dummy_result("English"), tmp)
            assert os.path.exists(os.path.join(tmp, "q1_results.json"))

    def test_output_dir_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "new_dir")
            save_results(_dummy_result("Arabic"), _dummy_result("English"), nested)
            assert os.path.isdir(nested)

    def test_arabic_gap_greater_marks_claim_supported(self):
        ar = _dummy_result("Arabic")   # gap=0.5, significant=True
        en = _dummy_result("English")  # gap=0.5
        ar["gap"] = 0.6
        with tempfile.TemporaryDirectory() as tmp:
            save_results(ar, en, tmp)
            with open(os.path.join(tmp, "q1_results.json"), encoding='utf-8') as f:
                data = json.load(f)
            assert data["comparative"]["claim_supported"] is True

    def test_arabic_gap_smaller_marks_claim_not_supported(self):
        ar = _dummy_result("Arabic")
        en = _dummy_result("English")
        ar["gap"] = 0.2  # less than english gap of 0.5
        with tempfile.TemporaryDirectory() as tmp:
            save_results(ar, en, tmp)
            with open(os.path.join(tmp, "q1_results.json"), encoding='utf-8') as f:
                data = json.load(f)
            assert data["comparative"]["claim_supported"] is False


# ── Lexicon sanity checks ──────────────────────────────────────────────────────

class TestLexicons:
    def test_arabic_families_non_empty(self):
        assert len(ARABIC_ROOT_FAMILIES) > 0

    def test_english_families_non_empty(self):
        assert len(ENGLISH_MORPH_FAMILIES) > 0

    def test_each_arabic_family_has_enough_words(self):
        for root, words in ARABIC_ROOT_FAMILIES.items():
            assert len(words) >= 3, f"Root {root} has fewer than 3 words"

    def test_each_english_family_has_enough_words(self):
        for stem, words in ENGLISH_MORPH_FAMILIES.items():
            assert len(words) >= 3, f"Stem {stem} has fewer than 3 words"

    def test_no_duplicate_words_within_arabic_family(self):
        for root, words in ARABIC_ROOT_FAMILIES.items():
            assert len(words) == len(set(words)), f"Duplicates in root {root}"

    def test_no_duplicate_words_within_english_family(self):
        for stem, words in ENGLISH_MORPH_FAMILIES.items():
            assert len(words) == len(set(words)), f"Duplicates in stem {stem}"
