# -*- coding: utf-8 -*-
"""
regenerate_v3.py — AL-MIR'ĀH coordinate space regeneration.

Brings the live engine into compliance with Paper B §3.1–3.2:
  - Embedding: THREE CARRIER SENTENCES, LAYER-8 hidden states,
    mean-pooled over the TARGET WORD's subword span, averaged across
    carriers. (The live v2 server used bare-term last-layer pooling —
    a paper/implementation mismatch discovered 2026-06-10.)
  - Basis re-fit: 99×99 cosine → Poincaré disk via stress descent,
    seed 42, replicating the Paper B §3.1 specification: similar
    Names close, repelled Names angularly opposed, tier banded by r
    (Dhāt inner, Ṣifāt mid, Afʿāl outer).
  - Outputs: poincare_data_v3.json (same schema as poincare_data.json),
    name_vecs_v3.npz (basis embeddings for the server), and a
    comparison report old-vs-new.

Implementation note to surface in Paper B prose: §3.1 says "average
the resulting layer-8 hidden states" without specifying the pooling
span. This script pools over the target word's subword tokens only
(carrier words excluded), which is the defensible reading; the paper
should state it explicitly.

Run:  python3 regenerate_v3.py [--names all_99_corrected.json]
                               [--old poincare_data.json]
                               [--terms terms.txt]   # optional re-projection list
"""

import json, re, argparse, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
LAYER = 8
SEED = 42

CARRIERS = [
    "الكلمة هي {w}",
    "يقول الشيخ {w}",
    "معنى {w} هو",
]

_DIAC = re.compile(r'[\u064B-\u065F\u0670\u0640]')


def strip_diac(t):
    return _DIAC.sub('', t)


# ----------------------------------------------------------------------
# Embedding — Paper B §3.1 protocol
# ----------------------------------------------------------------------

class CarrierEmbedder:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID,
                                               output_hidden_states=True)
        self.model.eval()

    @torch.no_grad()
    def embed(self, word: str) -> np.ndarray:
        w = strip_diac(word)
        vecs = []
        for c in CARRIERS:
            sent = c.format(w=w)
            enc = self.tok(sent, return_tensors="pt", truncation=True,
                           max_length=64)
            out = self.model(**enc)
            hid = out.hidden_states[LAYER][0]          # (seq, 768)
            span = self._word_span(sent, w, enc)
            vecs.append(hid[span].mean(0).numpy())
        return np.mean(vecs, axis=0)

    def _word_span(self, sent: str, w: str, enc) -> list:
        """Indices of the target word's subword tokens via offset mapping."""
        start = sent.index(w)
        end = start + len(w)
        enc2 = self.tok(sent, return_offsets_mapping=True, truncation=True,
                        max_length=64)
        idx = [i for i, (a, b) in enumerate(enc2["offset_mapping"])
               if a < end and b > start and (b - a) > 0]
        return idx if idx else list(range(1, len(enc2["offset_mapping"]) - 1))


# ----------------------------------------------------------------------
# Poincaré disk fit — Paper B §3.1 stress descent
# ----------------------------------------------------------------------

TIER_R = {0: 0.30, 1: 0.55, 2: 0.78}     # Dhāt / Ṣifāt / Afʿāl bands


def fit_disk(S: np.ndarray, levels: list, k_nn=5, k_rep=3,
             steps=500, lr=0.05, seed=SEED):
    """Angular stress descent with HARD tier-radius constraint.

    Paper B §3.1 places tier hierarchically by r; enforcing that as a
    constraint (not a soft penalty) and optimizing ANGLES only is both
    faithful to the spec and numerically stable: similar Names pulled
    to small angular separation, repelled Names pushed toward
    opposition (Δθ → π). Deterministic under seed."""
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    theta = rng.uniform(0, 2 * np.pi, n)
    r_fix = np.array([TIER_R[min(l, 2)] for l in levels])
    # deterministic small radial stagger inside each band (visual declutter)
    r_fix = r_fix + 0.04 * (rng.random(n) - 0.5)

    Sx = S.copy()
    np.fill_diagonal(Sx, -np.inf)
    nn = np.argsort(-Sx, axis=1)[:, :k_nn]
    nn_w = np.take_along_axis(S, nn, axis=1)
    np.fill_diagonal(Sx, np.inf)
    rep = np.argsort(Sx, axis=1)[:, :k_rep]

    for step in range(steps):
        g = np.zeros(n)
        for i in range(n):
            for jj, j in enumerate(nn[i]):
                w = max(nn_w[i, jj], 0.05)
                g[i] += w * np.sin(theta[i] - theta[j])          # attract
            for j in rep[i]:
                g[i] -= 0.6 * np.sin(theta[i] - theta[j])        # oppose
        theta -= lr * g
    P = np.stack([r_fix * np.cos(theta), r_fix * np.sin(theta)], axis=1)
    return P


def cos_matrix(V: np.ndarray) -> np.ndarray:
    Vc = V - V.mean(0)                     # vector centering, as in server
    Vn = Vc / (np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-12)
    return Vn @ Vn.T


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', default='all_99_corrected.json')
    ap.add_argument('--old', default='poincare_data.json')
    ap.add_argument('--terms', default=None,
                    help='optional newline list of accumulated terms to '
                         're-project after the fit')
    args = ap.parse_args()

    names_db = json.load(open(args.names, encoding='utf-8'))
    old = json.load(open(args.old, encoding='utf-8'))
    old_by_ar = {nd['ar']: nd for nd in old['nodes']}

    import os
    emb = None
    if os.path.exists('name_vecs_v3.npz'):
        z = np.load('name_vecs_v3.npz', allow_pickle=True)
        ars, V = list(z['ars']), z['V']
        levels = [old_by_ar.get(ar, {}).get('level', 1) for ar in ars]
        print('Loaded cached basis embeddings (name_vecs_v3.npz)')
    else:
        emb = CarrierEmbedder()
        print('Embedding 99 Names under carrier/layer-8 protocol...')
        ars, vecs, levels = [], [], []
        for nd in names_db:
            ar = nd['name_ar']
            ars.append(ar)
            vecs.append(emb.embed(ar))
            levels.append(old_by_ar.get(ar, {}).get('level', 1))
        V = np.stack(vecs)
        np.savez_compressed('name_vecs_v3.npz', ars=np.array(ars, dtype=object), V=V)
    if emb is None and args.terms:
        emb = CarrierEmbedder()

    S = cos_matrix(V)
    print(f'Similarity matrix: mean={S[np.triu_indices(99,1)].mean():.4f}, '
          f'std={S[np.triu_indices(99,1)].std():.4f}')

    P = fit_disk(S, levels)
    nodes = []
    for i, ar in enumerate(ars):
        px, py = float(P[i, 0]), float(P[i, 1])
        nodes.append({'ar': ar, 'px': round(px, 4), 'py': round(py, 4),
                      'r': round(float(np.hypot(px, py)), 4),
                      'level': levels[i]})
    json.dump({'nodes': nodes, 'meta': {
        'version': 'v3', 'embedding': f'carrier3/layer{LAYER}/word-span',
        'seed': SEED, 'model': MODEL_ID}},
        open('poincare_data_v3.json', 'w', encoding='utf-8'),
        ensure_ascii=False, indent=1)
    print('Wrote poincare_data_v3.json')

    # comparison report
    moves = []
    for nd in nodes:
        o = old_by_ar.get(nd['ar'])
        if o:
            o_r = float(np.hypot(o.get('px', 0.0), o.get('py', 0.0)))
            moves.append((nd['ar'],
                          float(np.hypot(nd['px'] - o['px'], nd['py'] - o['py'])),
                          o_r, nd['r']))
    moves.sort(key=lambda x: -x[1])
    print('\nLargest position changes (old→new):')
    for ar, d, ro, rn in moves[:10]:
        print(f'  {ar:14} Δ={d:.3f}  r: {ro:.2f}→{rn:.2f}')

    # tier-r separation check (the hyperbolic-depth hypothesis)
    rs = {0: [], 1: [], 2: []}
    for nd in nodes:
        rs[min(nd['level'], 2)].append(nd['r'])
    print('\nTier r-bands (mean ± std):')
    for lv, lab in [(0, 'Dhāt'), (1, 'Ṣifāt'), (2, 'Afʿāl')]:
        a = np.array(rs[lv])
        print(f'  {lab:6} n={len(a):2}  r={a.mean():.3f} ± {a.std():.3f}')

    if args.terms:
        terms = [t.strip() for t in open(args.terms, encoding='utf-8')
                 if t.strip()]
        print(f'\nRe-projecting {len(terms)} accumulated terms...')
        out = open('coordinates_v3.jsonl', 'w', encoding='utf-8')
        Vc_names = V - V.mean(0)
        Vn = Vc_names / (np.linalg.norm(Vc_names, axis=1, keepdims=True) + 1e-12)
        mean_vec = V.mean(0)
        for t in terms:
            v = emb.embed(t) - mean_vec
            v = v / (np.linalg.norm(v) + 1e-12)
            sims = Vn @ v
            order = np.argsort(-sims)
            top = [(ars[i], float(sims[i])) for i in order[:5]]
            w = sum(max(0, s) for _, s in top)
            px = sum(max(0, s) * nodes[i]['px'] for i, (_, s)
                     in zip(order[:5], top)) / w if w > 0 else 0.0
            py = sum(max(0, s) * nodes[i]['py'] for i, (_, s)
                     in zip(order[:5], top)) / w if w > 0 else 0.0
            out.write(json.dumps({
                'term_ar': t,
                'top_name_attractors': [[a, round(s, 4)] for a, s in top],
                'estimated_position': {'px': round(px, 4), 'py': round(py, 4),
                                       'r': round(float(np.hypot(px, py)), 4)},
                'pipeline': 'v3'}, ensure_ascii=False) + '\n')
        out.close()
        print('Wrote coordinates_v3.jsonl')


if __name__ == '__main__':
    main()
