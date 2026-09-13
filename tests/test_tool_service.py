"""MCP tool smoke tests without downloading a model or mutating a dataset."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from tool_service import ToolService


def result(term):
    return {
        "estimated_position": {"px": 0.2 if term == "رحمة" else 0.3,
                               "py": 0.1, "r": 0.25, "level": 1,
                               "level_label": "Ṣifāt"},
        "abjad": {"value": 12, "breakdown": "example"},
        "query_wazn": "?", "query_wazn_status": "ambiguous", "cluster_wazn": "فَعِيل",
        "dominant_wazn": "?", "poincare_dist_to_primary": 0.1,
        "top_names": [{"ar": "الرحمن", "trans": "Al-Raḥmān", "sim": 0.8,
                       "level": 1, "root": "ر-ح-م", "wazn": "فَعْلان",
                       "paired_opposite": "", "meaning": "mercy", "layer2_semantic": "mercy"}],
        "bottom_names": [{"ar": "المنتقم", "sim": -0.2}],
        "attractors_centered": [], "repelled_centered": [],
    }


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.name_meta = {
            "الرحمن": {"root": "ر-ح-م", "trans": "Al-Raḥmān", "wazn": "فَعْلان",
                       "level": 1, "abjad": 12, "meaning": "mercy",
                       "paired_opposite": "", "px": 0.2, "py": 0.1,
                       "ml_homolog": "an analogy"},
            "الرحيم": {"root": "ر-ح-م", "trans": "Al-Raḥīm", "wazn": "فَعِيل",
                       "level": 1, "abjad": 13, "meaning": "mercy",
                       "paired_opposite": "", "px": 0.3, "py": 0.1},
        }
        self.coord_db = [
            {"term_ar": "رحمة", "term_undiacritized": "رحمة",
             "estimated_position": {"px": 0.2, "py": 0.1, "r": 0.25,
                                    "level_label": "Ṣifāt"}},
            {"term_ar": "رأفة", "term_undiacritized": "رأفة",
             "estimated_position": {"px": 0.21, "py": 0.1, "r": 0.26,
                                    "level_label": "Ṣifāt"}},
        ]
        self.field_zero_source = {"value": "test"}

    def lookup(self, term):
        self.calls.append(f"lookup:{term}")
        return result(term)

    def profile(self, term):
        self.calls.append(f"profile:{term}")
        if term == "غير-مدعوم":
            raise ValueError("unsupported term")
        return {"term_ar": term, "position": {"r": 0.25}, "tier": "Ṣifāt",
                "dominant_wazn": "?", "abjad": {"value": 12},
                "attractors": [{"name_ar": "الرحمن", "paired_opposite": "المنتقم"}],
                "dataset_neighbors": [{"term_ar": "لطف"}]}

    def centroid(self, profiles):
        return (0.2, 0.1)

    def fit(self, profile, centroid):
        return {"رحمة": 0.9, "رأفة": 0.8}[profile["term_ar"]]

    def distance(self, left, right):
        return abs(left[0] - right[0])

    def compare_geometry(self, left, right):
        self.calls.append("geometry")
        return {"distance_euclidean": 0.1, "distance_hyperbolic": 0.3,
                "hierarchy_load": 3.0}

    def midpoint(self, left, right):
        return {"px": 0.25, "py": 0.1, "r": 0.27}

    def midpoint_names(self, px, py, k):
        return [{"ar": "الرحمن"}]

    def decomposition(self, left, right):
        return {"d_radial": 0.1, "d_angular": 0.2, "radial_share": 0.33,
                "angular_share": 0.67, "gloss": "field-dominated"}

    def root_geometry(self, positions):
        return {
            "karcher_mean": {"px": 0.25, "py": 0.1, "r": 0.27},
            "frechet_variance": 0.04, "dispersion": 0.2,
            "mean_pairwise": 0.4, "field_mean_pairwise": 0.8,
            "tightness": 0.5,
        }

    def strip(self, term):
        return term

    def abjad_breakdown(self, term):
        return "example"

    def abjad_value(self, term):
        return 13


class ToolServiceTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.service = ToolService(self.backend)

    def test_five_tool_contracts(self):
        cases = [
            ("philological_lookup", {"term": "رحمة"}, "PHILOLOGICAL COORDINATE"),
            ("root_analysis", {"root": "ر-ح-م"}, "ROOT ANALYSIS"),
            ("semantic_neighbors", {"term": "رحمة", "k": 2}, "رأفة"),
            ("compare_terms", {"term1": "رحمة", "term2": "رأفة"}, "hierarchy load"),
            ("semantic_project", {"candidates": {"mercy": ["رأفة", "رحمة"]},
                                  "context_arabic": ["رحمة"]}, "RECOMMENDED: رحمة"),
        ]
        for name, args, expected in cases:
            with self.subTest(name=name):
                self.assertIn(expected, self.service.execute(name, args))

    def test_comparison_always_looks_up_both_terms_before_geometry(self):
        text = self.service.execute("compare_terms", {"term1": "رحمة", "term2": "رأفة"})
        self.assertEqual(self.backend.calls[:3], ["lookup:رحمة", "lookup:رأفة", "geometry"])
        self.assertIn("Abjad (provisional)=12 [example]", text)
        self.assertIn("الرحمن (Al-Raḥmān, sim=0.8/0.8)", text)
        self.assertIn("nearest basis Names: الرحمن", text)

    def test_neighbor_unknown_term_looks_up_before_search(self):
        self.service.execute("semantic_neighbors", {"term": "لطف", "k": 1})
        self.assertEqual(self.backend.calls, ["lookup:لطف"])

    def test_invalid_inputs_do_not_call_backend(self):
        self.assertIn("Error", self.service.execute("compare_terms", {"term1": "رحمة"}))
        self.assertIn("Error", self.service.execute("semantic_neighbors",
                                                     {"term": "رحمة", "min_r": 0.9,
                                                      "max_r": 0.1}))
        self.assertEqual(self.backend.calls, [])

    def test_project_keeps_valid_candidate_after_one_failure(self):
        text = self.service.execute("semantic_project", {
            "candidates": {"mercy": ["غير-مدعوم", "رحمة"]},
            "context_arabic": ["غير-مدعوم", "رحمة"],
        })
        self.assertIn("غير-مدعوم: error", text)
        self.assertIn("RECOMMENDED: رحمة", text)

    def test_root_flags_provisional_abjad_mismatch(self):
        text = self.service.execute("root_analysis", {"root": "ر-ح-م"})
        self.assertIn("stored, provisional", text)
        self.assertIn("stored≠recomputed", text)
        self.assertIn("ml_homolog [framework-interpretive]", text)
        self.assertIn("Karcher mean: (0.25, 0.1) r=0.27", text)
        self.assertIn("Fréchet variance: 0.04", text)
        self.assertIn("Tightness ratio: 0.5", text)

    def test_project_without_context_does_not_recommend(self):
        text = self.service.execute("semantic_project", {
            "candidates": {"mercy": ["رحمة", "رأفة"]},
            "context_arabic": [],
        })
        self.assertIn("UNRANKED", text)
        self.assertIn("fit unavailable (no valid Arabic context)", text)
        self.assertIn("top attractor / axis: الرحمن ⇄ المنتقم", text)
        self.assertIn("nearest dataset neighbors: لطف", text)
        self.assertNotIn("RECOMMENDED", text)

    def test_project_with_context_shows_evidence_and_fit_status(self):
        text = self.service.execute("semantic_project", {
            "candidates": {"mercy": ["رأفة", "رحمة"]},
            "context_arabic": ["لطف"],
        })
        self.assertIn("fit=0.9", text)
        self.assertIn("top attractor / axis: الرحمن ⇄ المنتقم", text)
        self.assertIn("nearest dataset neighbors: لطف", text)
        self.assertIn("RECOMMENDED: رحمة (geometric fit=0.9; verify semantic sense)", text)


if __name__ == "__main__":
    unittest.main()
