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


def geodesic_midpoint(u, v):
    """Intrinsic midpoint of the geodesic segment from u to v."""
    u, v = _clip(u), _clip(v)
    if np.linalg.norm(u - v) < EPS:
        return u.copy()
    return exp_x(u, 0.5 * log_x(u, v))


def radial_angular_legs(u, v):
    """Return a diagnostic radial/angular L-path decomposition.

    The radial leg changes depth along u's ray until it reaches the
    shallower radius. The angular leg then connects the two angles at
    that radius. Their sum is an upper bound on the direct geodesic,
    not a unique orthogonal decomposition of that geodesic.
    """
    u, v = _clip(u), _clip(v)
    r1 = min(float(np.linalg.norm(u)), MAX_NORM)
    r2 = min(float(np.linalg.norm(v)), MAX_NORM)
    th1 = float(np.arctan2(u[1], u[0]))
    th2 = float(np.arctan2(v[1], v[0]))
    dth = abs(th1 - th2)
    if dth > np.pi:
        dth = 2.0 * np.pi - dth

    radial = abs(2.0 * np.arctanh(r1) - 2.0 * np.arctanh(r2))
    shallow = min(r1, r2)
    qa = np.array([shallow * np.cos(th1), shallow * np.sin(th1)])
    qb = np.array([shallow * np.cos(th2), shallow * np.sin(th2)])
    angular = poincare_dist(qa, qb)
    total = radial + angular
    return {
        "radial": float(radial),
        "angular": float(angular),
        "radial_share": None if total < EPS else float(radial / total),
        "angular_share": None if total < EPS else float(angular / total),
        "delta_theta_deg": float(np.degrees(dth)),
        "path_length": float(total),
        "geodesic_distance": poincare_dist(u, v),
    }


def mean_pairwise_distance(points):
    """Mean geodesic distance over all unordered point pairs."""
    pts = [_clip(p) for p in points]
    if len(pts) < 2:
        return 0.0
    distances = [
        poincare_dist(pts[i], pts[j])
        for i in range(len(pts))
        for j in range(i + 1, len(pts))
    ]
    return float(np.mean(distances))


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


def frechet_mean(points, weights=None, iters=256, tol=1e-9):
    """Robust intrinsic mean using backtracking Riemannian descent.

    This is used for new v3.2 cluster statistics. The historical
    ``karcher_mean`` iteration remains unchanged because it generated the
    frozen v3 query coordinates cited by Paper B.
    """
    pts = [_clip(p) for p in points]
    if not pts:
        raise ValueError("at least one point is required")
    if len(pts) == 1:
        return pts[0].copy()
    if len(pts) == 2 and weights is None:
        return geodesic_midpoint(pts[0], pts[1])

    if weights is None:
        w = np.ones(len(pts), dtype=float)
    else:
        w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if w.sum() < EPS:
        w = np.ones(len(pts), dtype=float)
    w /= w.sum()

    def objective(candidate):
        return sum(
            wi * poincare_dist(candidate, point) ** 2
            for wi, point in zip(w, pts)
        )

    x = _clip(sum(wi * point for wi, point in zip(w, pts)))
    for _ in range(iters):
        direction = sum(wi * log_x(x, point) for wi, point in zip(w, pts))
        if np.linalg.norm(direction) < tol:
            break
        current = objective(x)
        step = 1.0
        while step > 1e-8:
            candidate = exp_x(x, step * direction)
            if objective(candidate) < current - 1e-12:
                x = candidate
                break
            step *= 0.5
        else:
            break
    return x


def frechet_statistics(points, field_baseline=None):
    """Intrinsic center and dispersion statistics for disk points."""
    pts = [_clip(p) for p in points]
    if not pts:
        raise ValueError("at least one point is required")
    center = frechet_mean(pts)
    squared = [poincare_dist(center, p) ** 2 for p in pts]
    variance = float(np.mean(squared))
    pairwise = mean_pairwise_distance(pts)
    tightness = (
        None
        if field_baseline is None or field_baseline <= EPS
        else float(pairwise / field_baseline)
    )
    return {
        "center": center,
        "variance": variance,
        "dispersion": float(np.sqrt(variance)),
        "mean_pairwise": pairwise,
        "tightness": tightness,
    }


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
