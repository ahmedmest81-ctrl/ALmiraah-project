"""Compute v3.2 intrinsic diagnostics from saved v3 query records."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from hyperbolic import (  # noqa: E402
    geodesic_midpoint,
    poincare_dist,
    radial_angular_legs,
)


DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
TERMS = {
    "faraḥ": "فرح",
    "surūr": "سرور",
    "uḥibbuka": "أحبك",
    "aʿshaquki": "أعشقك",
    "ṣamt": "صمت",
    "ḥuzn": "حزن",
    "ghaḍab": "غضب",
    "khawf": "خوف",
    "fanāʾ": "فناء",
    "baqāʾ": "بقاء",
}
PAIRS = [
    ("faraḥ", "surūr", "paper_b_demonstration"),
    ("uḥibbuka", "aʿshaquki", "paper_b_demonstration"),
    ("ṣamt", "ḥuzn", "paper_b_demonstration"),
    ("ghaḍab", "khawf", "paper_b_demonstration"),
    ("fanāʾ", "baqāʾ", "paper_c_exploratory"),
]


def bare(text: str) -> str:
    return DIACRITICS.sub("", text or "")


def load_positions(path: Path) -> dict[str, dict]:
    wanted = {bare(term): label for label, term in TERMS.items()}
    positions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            label = wanted.get(bare(record.get("term_ar", "")))
            if label:
                positions[label] = record["estimated_position"]
    missing = sorted(set(TERMS) - set(positions))
    if missing:
        raise ValueError(f"Missing query records: {', '.join(missing)}")
    return positions


def nearest_names(
    point: np.ndarray,
    nodes: list[dict],
    transliterations: dict[str, str],
    k: int = 3,
) -> list[dict]:
    scored = sorted(
        (
            poincare_dist(point, np.array([node["px"], node["py"]])),
            node,
        )
        for node in nodes
    )
    return [
        {
            "name_ar": node["ar"],
            "name_trans": transliterations[node["ar"]],
            "distance": round(distance, 4),
        }
        for distance, node in scored[:k]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument(
        "--basis",
        type=Path,
        default=ROOT / "data" / "paper_b" / "basis_99_v3.json",
    )
    parser.add_argument(
        "--disk",
        type=Path,
        default=ROOT / "data" / "paper_b" / "poincare_data_v3.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    positions = load_positions(args.coordinates)
    basis = json.loads(args.basis.read_text(encoding="utf-8"))
    disk = json.loads(args.disk.read_text(encoding="utf-8"))
    transliterations = {
        entry["name_ar"]: entry["name_trans"] for entry in basis
    }

    results = []
    for term1, term2, status in PAIRS:
        p1 = np.array([
            positions[term1]["px"],
            positions[term1]["py"],
        ])
        p2 = np.array([
            positions[term2]["px"],
            positions[term2]["py"],
        ])
        midpoint = geodesic_midpoint(p1, p2)
        legs = radial_angular_legs(p1, p2)
        results.append({
            "term1": term1,
            "term1_ar": TERMS[term1],
            "term2": term2,
            "term2_ar": TERMS[term2],
            "status": status,
            "r1": round(float(np.linalg.norm(p1)), 4),
            "r2": round(float(np.linalg.norm(p2)), 4),
            "distance_hyperbolic": round(poincare_dist(p1, p2), 4),
            "radial_leg": round(legs["radial"], 4),
            "angular_leg": round(legs["angular"], 4),
            "radial_share": round(legs["radial_share"], 3),
            "angular_share": round(legs["angular_share"], 3),
            "delta_theta_deg": round(legs["delta_theta_deg"], 2),
            "midpoint": {
                "px": round(float(midpoint[0]), 4),
                "py": round(float(midpoint[1]), 4),
                "r": round(float(np.linalg.norm(midpoint)), 4),
                "nearest_basis_names": nearest_names(
                    midpoint,
                    disk["nodes"],
                    transliterations,
                ),
            },
        })

    payload = {
        "version": "v3.2",
        "query_dataset_commit": (
            "1aef46246bea66b64a460feec735ac001b1831e6"
        ),
        "basis_schema": "paper-b-basis-v3.0",
        "method_note": (
            "Radial/angular shares use an L-shaped diagnostic path at the "
            "shallower radius; they are not a unique decomposition of the "
            "direct geodesic."
        ),
        "pairs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
