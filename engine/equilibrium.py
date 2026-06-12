# -*- coding: utf-8 -*-
"""
equilibrium.py — Iʿtidāl correction at the measurement layer.

PROBLEM (live-confirmed 2026-06-10): a small set of basis Names
(Al-Ḥayy, Al-ʿAfuww, Al-Barr, Al-Ḥakam) appear as top attractors for
semantically unrelated query terms. This is CAMeLBERT-ca anisotropy:
the embedding cloud has a dominant common direction, so raw cosine
similarity to ANY probe is offset by each basis vector's alignment
with that common direction. The discriminative signal survives mostly
in the divergent/repelled sets — which is why the project's
"repelled Names are often more faithful" principle kept proving true.

FIX — and it is architecturally native to this project:
The Anchor Database doctrine states the zero-point is external and
constant, and every query is measured AGAINST it, not in raw terms.
The current pipeline violates its own doctrine at the measurement
layer: it reports raw cosine, i.e. distance from the origin of an
anisotropic cloud, not displacement from the field's equilibrium.

The correction: compute the field's zero μ (the mean similarity
profile each basis Name receives across a reference query set, or
across the basis itself), and report the CONTRASTIVE profile

    s̃_q[i] = s_q[i] − μ[i]

This is standard anisotropy correction in the embedding literature
(mean-centering / all-but-the-top, Mu & Viswanath 2018) and it is,
exactly, the iʿtidāl projection: each Name's pull on a query is
measured as deviation from that Name's equilibrium pull on the field.
A Name that pulls everything is, after centering, pulling nothing in
particular.

The basis itself is NOT modified. μ lives beside the basis, frozen,
recomputed only when the reference set is deliberately revised —
exactly the Anchor Database update discipline.
"""

import json
import numpy as np


# ----------------------------------------------------------------------
# Computing the field zero  μ
# ----------------------------------------------------------------------

def field_zero_from_queries(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    similarity_matrix: (n_queries, 99) raw cosine of accumulated queries
    against the basis (recoverable from the production JSONL, or by
    re-embedding the 722 accumulated terms).

    Returns μ ∈ R^99 — the mean pull each Name exerts on the field.
    Recommendation: compute over a CLEANED query set (see
    clean_dataset.py) so calibration noise (number-words, mojibake)
    does not contaminate the equilibrium.
    """
    return similarity_matrix.mean(axis=0)


def field_zero_from_basis(basis_similarity: np.ndarray) -> np.ndarray:
    """
    Fallback when no query set is available: μ from the 99×99 basis
    self-similarity matrix (excluding the diagonal). Less faithful to
    the production query distribution but requires nothing external.
    """
    S = basis_similarity.copy().astype(float)
    np.fill_diagonal(S, np.nan)
    return np.nanmean(S, axis=0)


# ----------------------------------------------------------------------
# Centered lookup
# ----------------------------------------------------------------------

def centered_profile(s_q: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """The iʿtidāl projection: deviation from the field's equilibrium."""
    return s_q - mu


def attractors_and_repelled(s_q: np.ndarray, mu: np.ndarray,
                            names: list, k_attr: int = 5,
                            k_rep: int = 3) -> dict:
    """
    Drop-in replacement for the raw-cosine top-k selection in
    philological_lookup. Returns BOTH raw and centered rankings so the
    server can expose them side by side during a validation period
    (backward compatibility for the accumulated dataset and Paper B §5
    values, which were computed on raw profiles).
    """
    c = centered_profile(s_q, mu)
    raw_order = np.argsort(-s_q)
    cen_order = np.argsort(-c)
    return {
        'attractors_raw':      [(names[i], float(s_q[i])) for i in raw_order[:k_attr]],
        'attractors_centered': [(names[i], float(c[i]))   for i in cen_order[:k_attr]],
        'repelled_raw':        [(names[i], float(s_q[i])) for i in raw_order[-k_rep:][::-1]],
        'repelled_centered':   [(names[i], float(c[i]))   for i in cen_order[-k_rep:][::-1]],
        'field_zero_norm':     float(np.linalg.norm(mu)),
        'note': ('attractors_centered measures each Name\'s pull as '
                 'deviation from its equilibrium pull on the field '
                 '(iʿtidāl projection / mean-centering anisotropy '
                 'correction). Hub Names that attract everything '
                 'rank low here unless they attract THIS term '
                 'specifically.'),
    }


# ----------------------------------------------------------------------
# Hub diagnostic — quantifies the problem before/after
# ----------------------------------------------------------------------

def hub_concentration(similarity_matrix: np.ndarray, names: list,
                      k: int = 5) -> list:
    """
    For each Name, the fraction of queries in which it appears among
    the top-k raw attractors. Run before deploying the correction to
    document the bug, and after (on centered profiles) to document
    the fix. Output belongs in the Paper B robustness appendix.
    """
    n_q = similarity_matrix.shape[0]
    counts = np.zeros(similarity_matrix.shape[1])
    for row in similarity_matrix:
        counts[np.argsort(-row)[:k]] += 1
    freq = counts / n_q
    order = np.argsort(-freq)
    return [(names[i], round(float(freq[i]), 4)) for i in order]


if __name__ == '__main__':
    # Synthetic sanity check: a hub direction contaminates raw top-k;
    # centering removes it.
    rng = np.random.default_rng(42)
    n_names, n_q = 99, 700
    hub = rng.normal(size=8)                       # common direction (8-dim toy)
    basis = rng.normal(size=(n_names, 8))
    basis[:4] += 2.5 * hub                          # four "hub Names"
    queries = rng.normal(size=(n_q, 8)) + 1.0 * hub  # all queries share the direction
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    S = queries @ basis.T
    names = [f'N{i}' for i in range(n_names)]
    mu = field_zero_from_queries(S)
    print('Top hub frequencies RAW:     ', hub_concentration(S, names)[:5])
    Sc = S - mu
    print('Top hub frequencies CENTERED:', hub_concentration(Sc, names)[:5])
