"""Deterministic tests for the v3.2 non-Euclidean operations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from hyperbolic import (  # noqa: E402
    exp_x,
    frechet_statistics,
    geodesic_midpoint,
    log_x,
    poincare_dist,
    radial_angular_legs,
)


class HyperbolicOperationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def point(self, max_radius: float = 0.9) -> np.ndarray:
        direction = self.rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        return direction * self.rng.uniform(0.0, max_radius)

    def test_exp_log_round_trip(self) -> None:
        for _ in range(200):
            x, y = self.point(0.8), self.point(0.8)
            recovered = exp_x(x, log_x(x, y))
            np.testing.assert_allclose(recovered, y, atol=1e-10)

    def test_midpoint_is_equidistant(self) -> None:
        for _ in range(500):
            x, y = self.point(), self.point()
            midpoint = geodesic_midpoint(x, y)
            left = poincare_dist(x, midpoint)
            right = poincare_dist(midpoint, y)
            self.assertAlmostEqual(left, right, places=10)
            self.assertAlmostEqual(left + right, poincare_dist(x, y), places=10)

    def test_radial_angular_diagnostic(self) -> None:
        radial = radial_angular_legs(
            np.array([0.2, 0.0]), np.array([0.7, 0.0])
        )
        self.assertAlmostEqual(radial["angular"], 0.0, places=10)
        self.assertAlmostEqual(radial["radial_share"], 1.0, places=10)

        angular = radial_angular_legs(
            np.array([0.5, 0.0]), np.array([0.0, 0.5])
        )
        self.assertAlmostEqual(angular["radial"], 0.0, places=10)
        self.assertAlmostEqual(angular["angular_share"], 1.0, places=10)

        for _ in range(500):
            result = radial_angular_legs(self.point(), self.point())
            self.assertGreaterEqual(
                result["path_length"] + 1e-10,
                result["geodesic_distance"],
            )

    def test_karcher_center_minimizes_sample_objective(self) -> None:
        points = [self.point(0.75) for _ in range(8)]
        stats = frechet_statistics(points)
        center = stats["center"]

        def objective(candidate: np.ndarray) -> float:
            return float(np.mean([
                poincare_dist(candidate, point) ** 2 for point in points
            ]))

        baseline = objective(center)
        for _ in range(2000):
            direction = self.rng.normal(size=2)
            direction /= np.linalg.norm(direction)
            tangent = direction * self.rng.uniform(1e-5, 0.04)
            perturbed = exp_x(center, tangent)
            self.assertGreaterEqual(objective(perturbed) + 1e-9, baseline)

    def test_frechet_center_handles_widely_separated_pair(self) -> None:
        left = np.array([-0.75, 0.15])
        right = np.array([0.72, -0.2])
        stats = frechet_statistics([left, right])
        center = stats["center"]
        self.assertAlmostEqual(
            poincare_dist(left, center),
            poincare_dist(center, right),
            places=10,
        )
        self.assertAlmostEqual(
            2.0 * stats["dispersion"],
            poincare_dist(left, right),
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
