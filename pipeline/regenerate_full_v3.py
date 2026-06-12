# -*- coding: utf-8 -*-
"""regenerate_full_v3.py — re-project the full accumulated dataset
under the v3 protocol, emitting server-schema records."""
import json, re, sys, time
import numpy as np
import torch
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, '.')
from wazn import parse_wazn
from hyperbolic import karcher_mean

MODEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
LAYER = 8
CARRIERS = ["الكلمة هي {w}", "يقول الشيخ {w}", "معنى {w} هو"]
_DIAC = re.compile(r'[\u064B-\u065F\u0670\u0640]')
ABJAD = {'ا':1,'أ':1,'إ':1,'آ':1,'ب':2,'ج':3,'د':4,'ه':5,'و':6,'ز':7,'ح':8,
         'ط':9,'ي':10,'ى':10,'ك':20,'ل':30,'م':40,'ن':50,'س':60,'ع':70,
         'ف':80,'ص':90,'ق':100,'ر':200,'ش':300,'ت':400,'ث':500,'خ':600,
         'ذ':700,'ض':800,'ظ':900,'غ':1000,'ة':400,'ؤ':6,'ئ':10}

strip_diac = lambda t: _DIAC.sub('', t)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, output_hidden_states=True)
model.eval()

def word_span(sent, w):
    s = sent.index(w); e = s + len(w)
    enc = tok(sent, return_offsets_mapping=True, truncation=True, max_length=64)
    idx = [i for i, (a, b) in enumerate(enc["offset_mapping"])
           if a < e and b > s and (b - a) > 0]
    return idx or list(range(1, len(enc["offset_mapping"]) - 1))

@torch.no_grad()
def embed(word):
    w = strip_diac(word)
    vs = []
    for c in CARRIERS:
        sent = c.format(w=w)
        enc = tok(sent, return_tensors="pt", truncation=True, max_length=64)
        out = model(**enc)
        vs.append(out.hidden_states[LAYER][0][word_span(sent, w)].mean(0).numpy())
    return np.mean(vs, axis=0)

# basis
z = np.load('name_vecs_v3.npz', allow_pickle=True)
ars, V = list(z['ars']), z['V']
nodes = {n['ar']: n for n in json.load(open('poincare_data_v3.json', encoding='utf-8'))['nodes']}
mean_vec = V.mean(0)
Vc = V - mean_vec
Vn = Vc / (np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-12)
LB = ["Dhāt", "Ṣifāt", "Afʿāl"]

src = open('coordinates_v2_full.jsonl.cleaned.jsonl', encoding='utf-8').readlines()
out = open('coordinates.jsonl', 'w', encoding='utf-8')
t0 = time.time()
for i, line in enumerate(src):
    old = json.loads(line)
    term = old['term_ar']
    try:
        v = embed(term) - mean_vec
        v = v / (np.linalg.norm(v) + 1e-12)
        sims = Vn @ v
        order = np.argsort(-sims)
        top = [(ars[j], float(sims[j])) for j in order[:5]]
        bot = [(ars[j], float(sims[j])) for j in order[-3:][::-1]]
        pts = [np.array([nodes[a]['px'], nodes[a]['py']]) for a, _ in top]
        ws = [max(0.0, s) for _, s in top]
        km = karcher_mean(pts, ws)
        px, py = float(km[0]), float(km[1])
        r = float(np.hypot(px, py))
        lv_votes = {}
        for a, s in top:
            lv = min(nodes[a]['level'], 2)
            lv_votes[lv] = lv_votes.get(lv, 0) + max(0, s)
        lv = max(lv_votes, key=lv_votes.get)
        wq = parse_wazn(term)
        rec = {
            "term_ar": term,
            "term_undiacritized": strip_diac(term),
            "top_name_attractors": [[a, round(s, 4)] for a, s in top],
            "bottom_names": [[a, round(s, 4)] for a, s in bot],
            "estimated_position": {"px": round(px, 4), "py": round(py, 4),
                                   "r": round(r, 4), "level": lv,
                                   "level_label": LB[lv]},
            "dominant_wazn": wq["query_wazn"] or "?",
            "query_wazn": wq["query_wazn"] or "?",
            "query_wazn_status": wq["query_wazn_status"],
            "abjad_value": sum(ABJAD.get(c, 0) for c in strip_diac(term)),
            "poincare_dist_to_primary": round(float(np.arccosh(max(1.0,
                1 + 2*((px-nodes[top[0][0]]['px'])**2 + (py-nodes[top[0][0]]['py'])**2) /
                max((1-px*px-py*py)*(1-nodes[top[0][0]]['px']**2-nodes[top[0][0]]['py']**2), 1e-9)))), 4),
            "timestamp": old.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "pipeline": "v3-carrier3-layer8-karcher",
        }
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f'  skip {term}: {e}')
    if (i + 1) % 100 == 0:
        el = time.time() - t0
        print(f'{i+1}/{len(src)}  ({el:.0f}s, {el/(i+1):.2f}s/term)')
out.close()
print(f'Done: {len(src)} terms in {time.time()-t0:.0f}s → coordinates.jsonl')
