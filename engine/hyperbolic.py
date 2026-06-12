# -*- coding: utf-8 -*-
"""
hyperbolic.py — Poincaré-disk geometry for AL-MIR'ĀH.

Coordinate-agnostic: operates on whatever fitted disk exists (v2 or
v3). Replaces the Euclidean shortcuts in the live engine with the
operations the hyperbolic model actually licenses:

  1. poincare_dist        — geodesic distance (the engine has this)
  2. mobius_add           — Möbius addition (gyrovector translation)
  3. exp0 / log0          — exponential/log maps at the origin
  4. karcher_mean         — TRUE weighted barycenter on the disk.
     The live engine interpolates positions as a EUCLIDEAN weighted
     centroid of attractor positions — which systematically biases
     query positions toward the disk's center, compressing r and
     contaminating any tier-from-r inference. The Karcher mean is the
     intrinsic replacement.
  5. hyperbolic_knn       — neighbor search by geodesic distance
  6. depth_statistics     — tests the hierarchy hypothesis: does r
     behave like depth (Dhāt < Ṣifāt < Afʿāl with real separation)?

References: Ungar (gyrovector spaces); Nickel & Kiela 2017 (Poincaré
embeddings); Karcher 1977 (Riemannian center of mass).
"""

import numpy as np

EPS = 1e-9
MAX_NORM = 1.0 - 1e-5


def _clip(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x)
    return x * (MAX_NORM / n) if n >= MAX_NORM else x


def poincare_dist(u, v):
    u, v = _clip(u), _clip(v)
    nu2, nv2 = u @ u, v @ v
    d2 = (u - v) @ (u - v)
    arg = 1.0 + 2.0 * d2 / max((1 - nu2) * (1 - nv2), EPS)
    return float(np.arccosh(max(1.0, arg)))


def mobius_add(u, v):
    """Möbius addition u ⊕ v on the disk (curvature −1)."""
    u, v = _clip(u), _clip(v)
    uv = u @ v
    nu2, nv2 = u @ u, v @ v
    num = (1 + 2 * uv + nv2) * u + (1 - nu2) * v
    den = 1 + 2 * uv + nu2 * nv2
    return _clip(num / max(den, EPS))


def exp0(t):
    """Exponential map at the origin: tangent vector → disk point."""
    t = np.asarray(t, dtype=float)
    n = np.linalg.norm(t)
    if n < EPS:
        return np.zeros_like(t)
    return _clip(np.tanh(n) * t / n)


def log0(p):
    """Log map at the origin: disk point → tangent vector."""
    p = _clip(p)
    n = np.linalg.norm(p)
    if n < EPS:
        return np.zeros_like(p)
    return np.arctanh(min(n, MAX_NORM)) * p / n


def mobius_neg(p):
    return -np.asarray(p, dtype=float)


def log_x(x, p):
    """Log map at x: direction/length of geodesic x→p in T_x."""
    q = mobius_add(mobius_neg(x), p)
    lam = 2.0 / max(1.0 - x @ x, EPS)
    n = np.linalg.norm(q)
    if n < EPS:
        return np.zeros_like(q)
    return (2.0 / lam) * np.arctanh(min(n, MAX_NORM)) * q / n


def exp_x(x, t):
    """Exponential map at x."""
    lam = 2.0 / max(1.0 - x @ x, EPS)
    n = np.linalg.norm(t)
    if n < EPS:
        return np.asarray(x, dtype=float)
    q = np.tanh(lam * n / 2.0) * t / n
    return mobius_add(x, q)


def karcher_mean(points, weights=None, iters=64, tol=1e-7):
    """
    Weighted Riemannian barycenter on the Poincaré disk via fixed-point
    iteration: x ← exp_x( Σ w_i log_x(p_i) / Σ w_i ).

    Drop-in replacement for the engine's Euclidean
    similarity-weighted centroid of attractor positions.
    """
    pts = [np.asarray(p, dtype=float) for p in points]
    if weights is None:
        w = np.ones(len(pts))
    else:
        w = np.asarray(weights, dtype=float)
        w = np.maximum(w, 0.0)
    if w.sum() < EPS:
        w = np.ones(len(pts))
    w = w / w.sum()

    # Euclidean init (fine as a starting point)
    x = _clip(sum(wi * pi for wi, pi in zip(w, pts)))
    for _ in range(iters):
        g = sum(wi * log_x(x, pi) for wi, pi in zip(w, pts))
        if np.linalg.norm(g) < tol:
            break
        x = exp_x(x, g)
    return x


def hyperbolic_knn(target, candidates, k=8):
    """candidates: list of (label, point). Returns k nearest by
    geodesic distance, with distances."""
    scored = [(lab, poincare_dist(target, p)) for lab, p in candidates]
    scored.sort(key=lambda x: x[1])
    return scored[:k]


def depth_statistics(nodes):
    """
    nodes: list of dicts with 'r' and 'level'. Tests whether r behaves
    as hierarchy depth. Returns per-tier stats + separation measures:
    pairwise band overlap and a rank-correlation of level with r.
    """
    from itertools import combinations
    by = {}
    for nd in nodes:
        by.setdefault(min(int(nd['level']), 2), []).append(float(nd['r']))
    stats = {lv: {'n': len(v), 'mean': float(np.mean(v)),
                  'std': float(np.std(v)),
                  'min': float(np.min(v)), 'max': float(np.max(v))}
             for lv, v in by.items()}
    # Spearman of level vs r (no scipy dependency)
    levels = np.array([min(int(nd['level']), 2) for nd in nodes], dtype=float)
    rs = np.array([float(nd['r']) for nd in nodes])
    def _rank(a):
        order = np.argsort(a)
        rk = np.empty_like(order, dtype=float)
        rk[order] = np.arange(len(a))
        return rk
    rl, rr = _rank(levels), _rank(rs)
    rho = float(np.corrcoef(rl, rr)[0, 1])
    overlaps = {}
    for a, b in combinations(sorted(by), 2):
        lo = max(stats[a]['min'], stats[b]['min'])
        hi = min(stats[a]['max'], stats[b]['max'])
        overlaps[f'{a}-{b}'] = max(0.0, hi - lo)
    return {'tiers': stats, 'spearman_level_r': rho,
            'band_overlap': overlaps}


if __name__ == '__main__':
    # sanity: Karcher vs Euclidean for points near the rim
    rng = np.random.default_rng(42)
    pts = [np.array([0.85, 0.0]), np.array([0.0, 0.85]),
           np.array([-0.85, 0.0])]
    euc = sum(pts) / 3
    kar = karcher_mean(pts)
    print('Euclidean centroid:', np.round(euc, 4), ' r =', round(float(np.hypot(*euc)), 4))
    print('Karcher mean:      ', np.round(kar, 4), ' r =', round(float(np.hypot(*kar)), 4))
    print('dist consistency (should be ~equal):',
          [round(poincare_dist(kar, p), 4) for p in pts])
