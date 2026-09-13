"""Paper A frequency-tier null invariants on a small deterministic fixture."""

import importlib.util
import random
import unittest
from collections import Counter
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "experiments" / "q2_robustness_freq_matched.py"
spec = importlib.util.spec_from_file_location("q2_frequency", MODULE)
q2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q2)


class FrequencyMatchedTest(unittest.TestCase):
    def test_orthographic_variants_share_base_value(self):
        self.assertEqual(q2.compute_word_abjad("أمّة", q2.MASHRIQI_VALUES),
                         q2.compute_word_abjad("امت", q2.MASHRIQI_VALUES))
        self.assertEqual(q2.normalize_letter("ـ"), "")

    def test_frequency_counts_unique_words_and_variant_glyphs(self):
        records = [{"word_i": "أب", "word_j": "اب"},
                   {"word_i": "أب", "word_j": "أب"}]
        counts = q2.compute_letter_frequencies(records)
        self.assertEqual(counts["ا"], 2)
        self.assertEqual(counts["ب"], 2)

    def test_permutation_preserves_each_frequency_tier(self):
        counts = Counter({letter: 28 - i for i, letter in enumerate(q2.MASHRIQI_VALUES)})
        for k in (4, 7):
            bins = q2.partition_letters_by_frequency(counts, k)
            self.assertEqual(sorted(letter for group in bins for letter in group),
                             sorted(q2.MASHRIQI_VALUES))
            self.assertEqual(max(map(len, bins)) - min(map(len, bins)), 0)
            shuffled = q2.make_freq_matched_permuted_values(bins, random.Random(42))
            for group in bins:
                self.assertEqual(
                    sorted(shuffled[letter] for letter in group),
                    sorted(q2.MASHRIQI_VALUES[letter] for letter in group),
                )
            self.assertTrue(any(shuffled[letter] != q2.MASHRIQI_VALUES[letter]
                                for letter in shuffled))

    def test_cross_root_selection_and_permutation_p_value(self):
        records = [
            {"word_i": "أب", "word_j": "جد", "root_relation": "different",
             "mean_attention": 0.2, "pos_distance": 1},
            {"word_i": "بج", "word_j": "ده", "root_relation": "same",
             "mean_attention": 0.9, "pos_distance": 2},
        ]
        calls = []
        original = q2.compute_partial_rho
        try:
            def fake_rho(abjad, attentions, pos, cooc):
                calls.append(len(attentions))
                return 0.5 if len(calls) == 1 else 0.1
            q2.compute_partial_rho = fake_rho
            summary = q2.run_freq_matched_null(records, {}, 3, 4, seed=42)
        finally:
            q2.compute_partial_rho = original
        self.assertEqual(calls, [1, 1, 1, 1])
        self.assertEqual(summary["empirical_p_two_tailed"], 0.25)
        self.assertFalse(summary["passed"])


if __name__ == "__main__":
    unittest.main()
