"""Model-independent operations behind the five AL-MIRʾĀH MCP tools.

The server supplies a backend; tests supply a small deterministic backend.
The service owns ordering, argument validation and the shared response shape,
while the MCP adapter only translates strings into protocol content blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence


class SemanticBackend(Protocol):
    name_meta: Mapping[str, Mapping[str, Any]]
    coord_db: Sequence[Mapping[str, Any]]
    field_zero_source: Mapping[str, str]

    def lookup(self, term: str) -> dict[str, Any]: ...
    def profile(self, term: str) -> dict[str, Any]: ...
    def centroid(self, profiles: list[dict[str, Any]]) -> tuple[float, float]: ...
    def fit(self, profile: dict[str, Any], centroid: tuple[float, float]) -> float: ...
    def distance(self, left: tuple[float, float], right: tuple[float, float]) -> float: ...
    def compare_geometry(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]: ...
    def midpoint(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]: ...
    def midpoint_names(self, px: float, py: float, k: int) -> list[dict[str, Any]]: ...
    def decomposition(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]: ...
    def root_geometry(self, positions: list[tuple[float, float]]) -> dict[str, Any]: ...
    def strip(self, term: str) -> str: ...
    def abjad_breakdown(self, term: str) -> str: ...
    def abjad_value(self, term: str) -> int: ...


@dataclass
class LiveBackend:
    """Typed dependency adapter for the research engine's existing functions."""

    name_meta: Mapping[str, Mapping[str, Any]]
    coord_db: Sequence[Mapping[str, Any]]
    field_zero_source: Mapping[str, str]
    lookup: Callable[[str], dict[str, Any]]
    profile: Callable[[str], dict[str, Any]]
    centroid: Callable[[list[dict[str, Any]]], tuple[float, float]]
    fit: Callable[[dict[str, Any], tuple[float, float]], float]
    distance: Callable[[tuple[float, float], tuple[float, float]], float]
    compare_geometry: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    midpoint: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    midpoint_names: Callable[[float, float, int], list[dict[str, Any]]]
    decomposition: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    root_geometry: Callable[[list[tuple[float, float]]], dict[str, Any]]
    strip: Callable[[str], str]
    abjad_breakdown: Callable[[str], str]
    abjad_value: Callable[[str], int]


TOOL_NAMES = (
    "philological_lookup", "root_analysis", "semantic_project",
    "semantic_neighbors", "compare_terms",
)


class ToolService:
    """Five-tool application service, independent of MCP and model startup."""

    def __init__(self, backend: SemanticBackend) -> None:
        self.backend = backend

    @staticmethod
    def _term(args: Mapping[str, Any], key: str) -> str:
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")
        return value.strip()

    def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        if name not in TOOL_NAMES:
            return f"Unknown tool: {name}"
        try:
            if name == "philological_lookup":
                return self.philological_lookup(self._term(arguments, "term"))
            if name == "root_analysis":
                return self.root_analysis(self._term(arguments, "root"))
            if name == "semantic_neighbors":
                return self.semantic_neighbors(
                    self._term(arguments, "term"), int(arguments.get("k", 8)),
                    float(arguments.get("min_r", 0.1)),
                    float(arguments.get("max_r", 0.95)),
                )
            if name == "compare_terms":
                return self.compare_terms(
                    self._term(arguments, "term1"), self._term(arguments, "term2")
                )
            candidates = arguments.get("candidates")
            context = arguments.get("context_arabic", [])
            if not isinstance(candidates, dict) or not candidates:
                raise ValueError("candidates must be a non-empty object")
            if not isinstance(context, list) or not all(isinstance(x, str) for x in context):
                raise ValueError("context_arabic must be a list of strings")
            if not all(isinstance(k, str) and isinstance(v, list) and
                       all(isinstance(x, str) for x in v) for k, v in candidates.items()):
                raise ValueError("candidate values must be lists of Arabic strings")
            return self.semantic_project(candidates, context)
        except (ValueError, TypeError) as exc:
            return f"Error: {exc}"

    def philological_lookup(self, term: str) -> str:
        result = self.backend.lookup(term)
        pos = result["estimated_position"]
        abjad = result.get("abjad", {})
        lines = [
            f"PHILOLOGICAL COORDINATE: {term}",
            f"Abjad value: {abjad.get('value', '—')} [{abjad.get('breakdown', '—')}]",
            f"Position: px={pos['px']}, py={pos['py']}, r={pos['r']}",
            f"Hierarchy: {pos['level_label']} (level {pos['level']})",
            f"Query wazn: {result['query_wazn']} [{result['query_wazn_status']}]",
            f"Cluster wazn (attractor majority): {result['cluster_wazn']}",
            f"Poincaré dist to primary: {result['poincare_dist_to_primary']}",
            "TOP ATTRACTOR NAMES (raw):",
        ]
        for item in result["top_names"]:
            lines.append(
                f"  {item['ar']} ({item['trans']}) sim={item['sim']} | "
                f"tier={item['level']} | root={item['root']} | wazn={item['wazn']} "
                f"| axis={item['paired_opposite']} | meaning={item['meaning']} "
                f"| {item['layer2_semantic']}"
            )
        lines.append(
            "IʿTIDĀL-CENTERED ATTRACTORS "
            f"({self.backend.field_zero_source['value']}): " +
            ", ".join(f"{n['ar']} ({n['sim_centered']:+.3f})"
                      for n in result.get("attractors_centered", []))
        )
        lines.append("IʿTIDĀL-CENTERED REPELLED: " +
                     ", ".join(n["ar"] for n in result.get("repelled_centered", [])))
        lines.append("STRUCTURALLY ABSENT (repelled, raw): " +
                     ", ".join(f"{n['ar']} ({n['sim']})" for n in result["bottom_names"]))
        return "\n".join(lines)

    def root_analysis(self, root: str) -> str:
        matches = [(ar, meta) for ar, meta in self.backend.name_meta.items()
                   if meta["root"] == root]
        if not matches:
            return f"No Names found with root {root}. Try format ر-ح-م."
        lines = [f"ROOT ANALYSIS: {root}"]
        for ar, meta in matches:
            bare = self.backend.strip(ar).removeprefix("ال")
            recomputed = self.backend.abjad_value(bare)
            tier = ("Dhāt", "Ṣifāt", "Afʿāl")[min(meta["level"], 2)]
            warning = (
                " ⚠ stored≠recomputed (pending Abjad audit)"
                if recomputed != meta["abjad"] else ""
            )
            lines.extend([
                f"{ar} ({meta['trans']}) | Wazn: {meta['wazn']} | Tier: {tier}",
                f"  Abjad (stored, provisional): {meta['abjad']} | "
                f"recomputed article-free: {self.backend.abjad_breakdown(bare)}{warning}",
                f"  Meaning: {meta['meaning']} | Axis: {meta['paired_opposite']}",
            ])
            for key in ("ml_homolog", "layer1_phonetic", "layer3_numerical",
                        "layer4_geometric", "layer5_breath"):
                if meta.get(key):
                    lines.append(f"  {key} [framework-interpretive]: {str(meta[key])[:100]}")
        if len(matches) > 1:
            geo = self.backend.root_geometry(
                [(meta["px"], meta["py"]) for _, meta in matches]
            )
            center = geo["karcher_mean"]
            tightness = geo["tightness"]
            reading = (
                "tighter than the field at large"
                if tightness is not None and tightness < 1.0
                else "no tighter than the field at large"
            )
            lines.extend([
                f"DISK GEOMETRY ({len(matches)} Names on root {root})",
                f"  Karcher mean: ({center['px']}, {center['py']}) r={center['r']}",
                f"  Fréchet variance: {geo['frechet_variance']} | "
                f"dispersion: {geo['dispersion']}",
                f"  Mean pairwise geodesic: {geo['mean_pairwise']} | "
                f"field baseline (all 99): {geo['field_mean_pairwise']}",
                f"  Tightness ratio: {tightness} ({reading})",
            ])
        return "\n".join(lines)

    def semantic_neighbors(self, term: str, k: int, min_r: float, max_r: float) -> str:
        if k < 1 or min_r < 0 or max_r >= 1 or min_r > max_r:
            raise ValueError("k must be positive and 0 ≤ min_r ≤ max_r < 1")
        if not self.backend.coord_db:
            return "Dataset not loaded — neighbors unavailable"
        clean = self.backend.strip(term)
        target = next((e for e in self.backend.coord_db
                       if e.get("term_undiacritized") == clean or
                       self.backend.strip(e.get("term_ar", "")) == clean), None)
        # A known coordinate is retrieved; an unknown term is looked up first.
        pos = (
            target["estimated_position"] if target
            else self.backend.lookup(term)["estimated_position"]
        )
        point = (pos["px"], pos["py"])
        distances: list[tuple[float, Mapping[str, Any]]] = []
        seen: set[str] = set()
        for entry in self.backend.coord_db:
            key = entry.get("term_undiacritized") or self.backend.strip(entry.get("term_ar", ""))
            if key == clean or key in seen:
                continue
            seen.add(key)
            p = entry.get("estimated_position", {})
            if min_r <= p.get("r", 0) <= max_r:
                distance = self.backend.distance(point, (p.get("px", 0), p.get("py", 0)))
                distances.append((distance, entry))
        lines = [f"SEMANTIC NEIGHBORS: {target.get('term_ar', term) if target else term}",
                 f"Position: {point} | {'found in dataset' if target else 'computed on-the-fly'}",
                 f"Filters: min_r={min_r}, max_r={max_r} | Top {k}"]
        for i, (distance, entry) in enumerate(sorted(distances, key=lambda x: x[0])[:k], 1):
            p = entry["estimated_position"]
            names = ", ".join(a[0] for a in entry.get("top_name_attractors", [])[:3])
            lines.append(f"{i}. {entry['term_ar']} | dist={distance:.4f} | "
                         f"r={p['r']} | {p.get('level_label', '')} | attractors: {names}")
        return "\n".join(lines)

    def semantic_project(self, candidates: Mapping[str, list[str]], context: list[str]) -> str:
        context_profiles = []
        for term in context:
            if term.strip():
                try:
                    context_profiles.append(self.backend.profile(term.strip()))
                except Exception:
                    # One unsupported context word must not discard all candidates.
                    continue
        center = self.backend.centroid(context_profiles)
        lines = ["SEMANTIC PROJECTION", f"Context: {', '.join(context)}", f"Centroid: {center}"]
        for concept, forms in candidates.items():
            lines.append(f"CONCEPT: {concept}")
            profiles = []
            for form in forms:
                if form.strip():
                    try:
                        profile = self.backend.profile(form.strip())
                        profile["fit_score"] = (
                            self.backend.fit(profile, center) if context_profiles else None
                        )
                        profiles.append(profile)
                    except Exception as exc:
                        lines.append(f"  {form}: error — {exc}")
            if context_profiles:
                profiles.sort(key=lambda p: p["fit_score"], reverse=True)
            for profile in profiles:
                attrs = profile["attractors"]
                top = attrs[0] if attrs else {}
                fit = (
                    f"fit={profile['fit_score']}"
                    if profile["fit_score"] is not None
                    else "fit unavailable (no valid Arabic context)"
                )
                abjad = profile.get("abjad", {}).get("value", "—")
                lines.append(
                    f"  {profile['term_ar']} | {fit} | r={profile['position']['r']} "
                    f"| {profile['tier']} | wazn={profile['dominant_wazn']} "
                    f"| Abjad (provisional)={abjad}"
                )
                lines.append(
                    f"    top attractor / axis: {top.get('name_ar', '—')} ⇄ "
                    f"{str(top.get('paired_opposite', '—'))[:40]}"
                )
                neighbors = profile.get("dataset_neighbors", [])
                lines.append(
                    "    nearest dataset neighbors: " +
                    (", ".join(n["term_ar"] for n in neighbors[:3]) if neighbors else "none")
                )
            if profiles and context_profiles:
                lines.append(
                    f"  → RECOMMENDED: {profiles[0]['term_ar']} "
                    f"(geometric fit={profiles[0]['fit_score']}; verify semantic sense)"
                )
            elif profiles:
                lines.append(
                    "  UNRANKED: no valid Arabic context; candidate senses remain ambiguous"
                )
        return "\n".join(lines)

    def compare_terms(self, left: str, right: str) -> str:
        # Lookup precedes all comparison operations, including midpoint names.
        first, second = self.backend.lookup(left), self.backend.lookup(right)
        a, b = first["estimated_position"], second["estimated_position"]
        distances = self.backend.compare_geometry(a, b)
        midpoint = self.backend.midpoint(a, b)
        names = self.backend.midpoint_names(midpoint["px"], midpoint["py"], 3)
        legs = self.backend.decomposition(a, b)
        top_a = {x["ar"]: x for x in first["top_names"]}
        top_b = {x["ar"]: x for x in second["top_names"]}
        bot_a = {x["ar"] for x in first["bottom_names"]}
        bot_b = {x["ar"] for x in second["bottom_names"]}
        shared = sorted(top_a.keys() & top_b.keys())
        opposite = sorted((top_a.keys() & bot_b) | (top_b.keys() & bot_a))
        only_a = sorted(top_a.keys() - top_b.keys())
        only_b = sorted(top_b.keys() - top_a.keys())
        abjad_a, abjad_b = first.get("abjad", {}), second.get("abjad", {})
        midpoint_names = ", ".join(
            f"{n['ar']} ({n.get('trans', '—')}, {n.get('tier', '—')}, "
            f"d={n.get('distance', '—')})" for n in names
        ) or "none"
        shared_text = ", ".join(
            f"{ar} ({top_a[ar]['trans']}, sim={top_a[ar]['sim']}/{top_b[ar]['sim']})"
            for ar in shared
        ) or "none"
        only_a_text = ", ".join(
            f"{ar} ({top_a[ar]['trans']}, sim={top_a[ar]['sim']})" for ar in only_a
        ) or "none"
        only_b_text = ", ".join(
            f"{ar} ({top_b[ar]['trans']}, sim={top_b[ar]['sim']})" for ar in only_b
        ) or "none"
        return "\n".join([
            f"COMPARISON: {left} ↔ {right}",
            f"{left}: r={a['r']}, {a['level_label']}, wazn={first['dominant_wazn']}, "
            f"Abjad (provisional)={abjad_a.get('value', '—')} "
            f"[{abjad_a.get('breakdown', '—')}]",
            f"{right}: r={b['r']}, {b['level_label']}, wazn={second['dominant_wazn']}, "
            f"Abjad (provisional)={abjad_b.get('value', '—')} "
            f"[{abjad_b.get('breakdown', '—')}]",
            f"Distance — Euclidean (flat): {distances['distance_euclidean']} | "
            f"hyperbolic (geodesic): {distances['distance_hyperbolic']} | "
            f"hierarchy load: {distances['hierarchy_load']}",
            f"DECOMPOSITION: radial={legs['d_radial']} | angular={legs['d_angular']} | "
            f"shares={legs['radial_share']}/{legs['angular_share']} | "
            f"Δθ={legs.get('delta_theta_deg', '—')}° | {legs['gloss']}",
            f"BARZAKH (geodesic midpoint): {midpoint} | "
            f"nearest basis Names: {midpoint_names}",
            f"SHARED ATTRACTORS: {shared_text}",
            f"DIVERGENT — {left} only: {only_a_text}",
            f"DIVERGENT — {right} only: {only_b_text}",
            f"OPPOSING POLES: {', '.join(opposite) or 'none'}",
        ])
