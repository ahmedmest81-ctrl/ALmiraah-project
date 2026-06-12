"""
AL-MIR'ĀH · Philological Query Engine + MCP Server
HuggingFace Space — WELLyes1/almiraah_transformer

v2.0 changes:
  - MCP transport: SSE → Streamable HTTP (mcp spec 2025-03-26)
  - save_to_dataset: embedding vectors stripped, deduplication added
  - philological_lookup: Abjad breakdown + axis label added to output
  - root_analysis: six-layer fields surfaced where available
  - Endpoints unchanged: /query /neighbors /compare /project /health
"""

import json, re, os, logging, asyncio
from datetime import datetime, timezone
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from huggingface_hub import HfApi
from mcp.server import Server
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp import types

# ── v2.1 fix modules (wazn inheritance bug; see DEPLOYMENT.md) ────────
import wazn as wazn_mod

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("almiraah")

MODEL_ID   = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
DATASET_ID = "WELLyes1/almiraah_coordinate_db"
HF_TOKEN   = os.environ.get("HF_TOKEN", None)
TOP_N      = 5

# ── Mashriqi Abjad values ─────────────────────────────────────────────
ABJAD = {
    'ا':1,'أ':1,'إ':1,'آ':1,'ب':2,'ج':3,'د':4,'ه':5,'و':6,'ز':7,
    'ح':8,'ط':9,'ي':10,'ى':10,'ك':20,'ل':30,'م':40,'ن':50,'س':60,
    'ع':70,'ف':80,'ص':90,'ق':100,'ر':200,'ش':300,'ت':400,'ث':500,
    'خ':600,'ذ':700,'ض':800,'ظ':900,'غ':1000,'ة':400,'ؤ':6,'ئ':10,
}

def abjad_value(text: str) -> int:
    return sum(ABJAD.get(c, 0) for c in strip_diac(text))

def abjad_breakdown(text: str) -> str:
    clean = strip_diac(text)
    parts = [f"{c}={ABJAD.get(c,0)}" for c in clean if c in ABJAD]
    total = sum(ABJAD.get(c, 0) for c in clean)
    return " + ".join(parts) + f" = {total}"

# ── Load model ────────────────────────────────────────────────────────
log.info("Loading CAMeLBERT-ca...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, output_hidden_states=True)
model.eval()
log.info("Model loaded.")

# ── v3 embedding protocol: 3 carrier sentences, layer-8, word-span ───
# Brings the server into compliance with Paper B §3.1–3.2. The v2
# bare-term/last-layer pooling is retired; coordinates produced under
# it live in coordinates_v2_*.jsonl archives.
EMB_LAYER = 8
CARRIERS = ["الكلمة هي {w}", "يقول الشيخ {w}", "معنى {w} هو"]

# ── Load data ─────────────────────────────────────────────────────────
with open("all_99_corrected.json", encoding="utf-8") as f:
    names_db = json.load(f)

# v3 disk fit preferred when shipped alongside
_POINCARE_FILE = ("poincare_data_v3.json"
                  if os.path.exists("poincare_data_v3.json")
                  else "poincare_data.json")
with open(_POINCARE_FILE, encoding="utf-8") as f:
    poincare_raw = json.load(f)
log.info(f"Disk fit: {_POINCARE_FILE}")

poincare_by_ar = {n["ar"]: n for n in poincare_raw["nodes"]}

# ── Load coordinates dataset into memory (for /neighbors) ─────────────
coord_db = []
coord_db_terms = set()   # fast dedup lookup

def load_coord_db():
    global coord_db, coord_db_terms
    entries = []
    if os.path.exists("coordinates.jsonl"):
        with open("coordinates.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
        log.info(f"Loaded {len(entries)} entries from local coordinates.jsonl")
    elif HF_TOKEN:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=DATASET_ID,
                filename="coordinates.jsonl",
                repo_type="dataset",
                token=HF_TOKEN,
            )
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            pass
            log.info(f"Loaded {len(entries)} entries from HF dataset")
        except Exception as e:
            log.warning(f"Could not load coordinates from HF dataset: {e}")
    else:
        log.warning("No HF_TOKEN and no local coordinates.jsonl — /neighbors disabled")
    coord_db = [e for e in entries if not _is_noise_record(e)]
    n_dropped = len(entries) - len(coord_db)
    if n_dropped:
        log.info(f"Hygiene filter dropped {n_dropped} noise records "
                 f"(number-words / digits / mojibake)")
    coord_db_terms = {
        e.get("term_undiacritized") or re.sub(r'[\u064B-\u065F\u0670\u0640]', '', e.get("term_ar", ""))
        for e in coord_db
    }

# ── v2.1 hygiene filter (see clean_dataset.py for the full offline pass) ──
_NOISE_TERMS = {
    'واحد','اثنان','اثنين','ثلاثة','أربعة','خمسة','ستة','سبعة','ثمانية',
    'تسعة','عشرة','عشرون','ثلاثون','أربعون','خمسون','ستون','سبعون',
    'ثمانون','تسعون','مائة','مائتان','مئة','مئتان','ألف','ألفان','مليون',
}
_MOJIBAKE_CHARS = set('ÃÂØÙÐ')

def _is_noise_record(e: dict) -> bool:
    t = (e.get("term_ar") or "").strip()
    bare = re.sub(r'^(ال)', '', re.sub(r'[\u064B-\u065F\u0670\u0640]', '', t))
    if bare in _NOISE_TERMS or t in _NOISE_TERMS:
        return True
    if re.search(r'[0-9\u0660-\u0669]', t):
        return True
    if any(c in _MOJIBAKE_CHARS for c in t):
        return True
    return False

load_coord_db()

# ── Core helpers ──────────────────────────────────────────────────────
def strip_diac(t):
    return re.sub(r'[\u064B-\u065F\u0670\u0640]', '', t)

def poincare_distance(u, v):
    ux, uy = u
    vx, vy = v
    u_arr = np.array([ux, uy])
    v_arr = np.array([vx, vy])
    du2 = min(float(np.dot(u_arr, u_arr)), 0.9999)
    dv2 = min(float(np.dot(v_arr, v_arr)), 0.9999)
    diff2 = float(np.dot(u_arr - v_arr, u_arr - v_arr))
    arg = 1.0 + 2.0 * diff2 / ((1.0 - du2) * (1.0 - dv2))
    return float(np.arccosh(max(1.0, arg)))

def _word_span(sent, w):
    start = sent.index(w); end = start + len(w)
    enc = tok(sent, return_offsets_mapping=True, truncation=True, max_length=64)
    idx = [i for i, (a, b) in enumerate(enc["offset_mapping"])
           if a < end and b > start and (b - a) > 0]
    return idx if idx else list(range(1, len(enc["offset_mapping"]) - 1))

def embed(text):
    """v3: carrier sentences, layer-8 hidden states, target-word span
    pooling, averaged across carriers (Paper B §3.1)."""
    w = strip_diac(text)
    vecs = []
    for c in CARRIERS:
        sent = c.format(w=w)
        inputs = tok(sent, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = model(**inputs)
        hid = out.hidden_states[EMB_LAYER][0]
        vecs.append(hid[_word_span(sent, w)].mean(0).numpy())
    return np.mean(vecs, axis=0)

# ── Pre-embed all 99 Names (cached v3 embeddings preferred) ──────────
log.info("Pre-embedding 99 Names...")
_cached_vecs = {}
if os.path.exists("name_vecs_v3.npz"):
    _z = np.load("name_vecs_v3.npz", allow_pickle=True)
    _cached_vecs = {ar: v for ar, v in zip(list(_z["ars"]), _z["V"])}
    log.info(f"Loaded {len(_cached_vecs)} cached v3 basis embeddings")
name_vecs = {}
name_meta = {}
for nd in names_db:
    ar = nd["name_ar"]
    name_vecs[ar] = _cached_vecs.get(ar)
    if name_vecs[ar] is None:
        name_vecs[ar] = embed(ar)
    po = poincare_by_ar.get(ar, {})
    name_meta[ar] = {
        "trans":           nd["name_trans"],
        "meaning":         nd["name_meaning"],
        "root":            nd["root"],
        "wazn":            nd["wazn"],
        "abjad":           nd["abjad_num"],
        "level":           po.get("level", 1),
        "px":              po.get("px", 0.0),
        "py":              po.get("py", 0.0),
        "layer2_semantic": nd.get("layer2_semantic", ""),
        "paired_opposite": nd.get("paired_opposite", ""),
        "ml_homolog":      nd.get("ml_homolog", ""),
        "layer1_phonetic": nd.get("layer1_phonetic", ""),
        "layer3_numerical":nd.get("layer3_numerical", ""),
        "layer4_geometric":nd.get("layer4_geometric", ""),
        "layer5_breath":   nd.get("layer5_breath", ""),
    }

all_vecs    = np.array(list(name_vecs.values()))
global_mean = all_vecs.mean(axis=0)

# ── v2.1 iʿtidāl: profile-space field zero μ ──────────────────────────
# cos_sim already centers VECTORS against the basis centroid; this is
# the complementary PROFILE centering: μ[Name] = the mean similarity
# that Name receives across the basis field. A query's centered pull
# toward a Name is sim − μ[Name]: deviation from the Name's
# equilibrium pull, not raw alignment. Fixes hub-attractor
# concentration (Al-Ḥayy / Al-ʿAfuww dominating unrelated lookups).
def _cos_raw_centered(a, b):
    a = a - global_mean; b = b - global_mean
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0

_name_list = list(name_vecs.keys())
FIELD_ZERO = {}
for _ar in _name_list:
    _vals = [_cos_raw_centered(name_vecs[_ar], name_vecs[_b])
             for _b in _name_list if _b != _ar]
    FIELD_ZERO[_ar] = float(np.mean(_vals))
log.info(f"Field zero μ computed over basis "
         f"(min={min(FIELD_ZERO.values()):.4f}, "
         f"max={max(FIELD_ZERO.values()):.4f})")

# ── v2.1b: upgrade μ to the QUERY distribution in the background ──────
# The basis μ above is a fallback: hubness manifests against general
# query vocabulary, not against the other 98 Names (live-verified
# 2026-06-10: basis μ left Al-Ḥayy rankings unchanged). Once the
# accumulated (hygiene-filtered) query terms are re-embedded, μ is
# swapped atomically to the query-distribution version. Until then the
# server runs on the basis fallback. Cap + deterministic sampling keep
# the cost bounded on CPU Spaces.
FIELD_ZERO_SOURCE = {"value": "basis_fallback"}
_MU_QUERY_CAP = int(os.environ.get("FIELD_ZERO_QUERY_CAP", "600"))

def _compute_query_mu():
    global FIELD_ZERO
    try:
        terms = []
        seen = set()
        for e in coord_db:
            t = (e.get("term_ar") or "").strip()
            k = e.get("term_undiacritized") or strip_diac(t)
            if t and k not in seen:
                seen.add(k)
                terms.append(t)
        if len(terms) < 30:
            log.info("Query-μ skipped: too few accumulated terms")
            return
        if len(terms) > _MU_QUERY_CAP:
            import random
            rnd = random.Random(42)
            terms = rnd.sample(terms, _MU_QUERY_CAP)
        sums = {ar: 0.0 for ar in name_vecs}
        n = 0
        for t in terms:
            try:
                v = embed(t)
                for ar, nv in name_vecs.items():
                    sums[ar] += cos_sim(v, nv)
                n += 1
            except Exception:
                continue
        if n >= 30:
            FIELD_ZERO = {ar: s / n for ar, s in sums.items()}
            FIELD_ZERO_SOURCE["value"] = f"query_distribution(n={n})"
            log.info(f"Field zero μ upgraded to query distribution "
                     f"(n={n}, min={min(FIELD_ZERO.values()):.4f}, "
                     f"max={max(FIELD_ZERO.values()):.4f})")
    except Exception as e:
        log.warning(f"Query-μ computation failed, basis fallback kept: {e}")

import threading
threading.Thread(target=_compute_query_mu, daemon=True).start()

log.info(f"Ready. {len(name_vecs)} Names embedded.")

# ── HuggingFace dataset save (dedup + no embeddings) ─────────────────
hf_api = HfApi(token=HF_TOKEN) if HF_TOKEN else None

def save_to_dataset(record: dict):
    """
    Append record to coordinates.jsonl in HF dataset.
    - Skips if term already in coord_db (deduplication).
    - Never saves embedding vectors (storage control).
    """
    global coord_db, coord_db_terms

    term_key = record.get("term_undiacritized", strip_diac(record.get("term_ar", "")))
    if term_key in coord_db_terms:
        return  # already stored — skip

    # Strip embedding if accidentally included
    record.pop("embedding", None)

    # Update in-memory db
    coord_db.append(record)
    coord_db_terms.add(term_key)

    if not hf_api:
        log.warning("HF_TOKEN not set — query not saved.")
        return
    try:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        try:
            existing = hf_api.hf_hub_download(
                repo_id=DATASET_ID, filename="coordinates.jsonl",
                repo_type="dataset", token=HF_TOKEN,
            )
            with open(existing, encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = ""
        content += line
        hf_api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo="coordinates.jsonl",
            repo_id=DATASET_ID, repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"add: {record['term_ar']}",
        )
        log.info(f"Saved: {record['term_ar']}")
    except Exception as e:
        log.error(f"Dataset write failed: {e}")

# ── Similarity helpers ────────────────────────────────────────────────
def cos_sim(a, b):
    a = a - global_mean; b = b - global_mean
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0

def poincare_dist(u, v):
    u, v = np.array(u), np.array(v)
    nu2  = np.dot(u, u); nv2 = np.dot(v, v)
    nd2  = np.dot(u - v, u - v)
    denom = (1 - nu2) * (1 - nv2)
    return float(np.arccosh(max(1.0, 1 + 2 * nd2 / denom))) if denom > 1e-10 else 9.99

# ── Core query logic (shared by HTTP + MCP) ───────────────────────────
def run_query(term: str) -> dict:
    vec  = embed(term)
    sims = sorted(
        [(ar, cos_sim(vec, nv)) for ar, nv in name_vecs.items()],
        key=lambda x: -x[1]
    )
    top    = sims[:TOP_N]
    bottom = sims[-3:]

    w_total = sum(max(0, s) for _, s in top)
    if w_total > 0:
        # v3: TRUE hyperbolic barycenter (Karcher mean). The previous
        # Euclidean weighted centroid compressed r by ~0.03 on average
        # and flipped tier assignment for ~20% of validation terms.
        from hyperbolic import karcher_mean
        _pts = [np.array([name_meta[ar]["px"], name_meta[ar]["py"]])
                for ar, _ in top]
        _ws  = [max(0, s) for _, s in top]
        _km  = karcher_mean(_pts, _ws)
        est_px, est_py = float(_km[0]), float(_km[1])
    else:
        est_px = est_py = 0.0
    est_r = float(np.sqrt(est_px**2 + est_py**2))

    lv_votes = {}
    for ar, s in top:
        lv = name_meta[ar]["level"]
        lv_votes[lv] = lv_votes.get(lv, 0) + max(0, s)
    est_level = max(lv_votes, key=lv_votes.get) if lv_votes else 1

    level_labels = ["Dhāt", "Ṣifāt", "Afʿāl"]

    wz_votes = {}
    for ar, s in top:
        w = name_meta[ar]["wazn"]
        wz_votes[w] = wz_votes.get(w, 0) + max(0, s)
    cluster_wazn = max(wz_votes, key=wz_votes.get) if wz_votes else "?"

    # v2.1 fix: parse the QUERY TERM's own wazn (never inherit cluster's)
    _wq = wazn_mod.parse_wazn(term)
    query_wazn        = _wq["query_wazn"] or "?"
    query_wazn_status = _wq["query_wazn_status"]

    # v2.1 iʿtidāl: profile-centered rankings alongside raw
    centered = sorted([(ar, s - FIELD_ZERO[ar]) for ar, s in sims],
                      key=lambda x: -x[1])
    top_centered    = centered[:TOP_N]
    bottom_centered = centered[-3:]

    primary_pos = (name_meta[top[0][0]]["px"], name_meta[top[0][0]]["py"])
    p_dist = poincare_dist([est_px, est_py], primary_pos)

    # Abjad value and breakdown for queried term
    term_abjad       = abjad_value(term)
    term_abjad_break = abjad_breakdown(term)

    result = {
        "term": term,
        "abjad": {
            "value":     term_abjad,
            "breakdown": term_abjad_break,
        },
        "top_names": [
            {
                "ar":              ar,
                "sim":             round(s, 4),
                "trans":           name_meta[ar]["trans"],
                "level":           name_meta[ar]["level"],
                "meaning":         name_meta[ar]["meaning"],
                "layer2_semantic": name_meta[ar]["layer2_semantic"][:120],
                "paired_opposite": name_meta[ar]["paired_opposite"][:80],
                "root":            name_meta[ar]["root"],
                "abjad":           name_meta[ar]["abjad"],
                "wazn":            name_meta[ar]["wazn"],
            }
            for ar, s in top
        ],
        "bottom_names": [
            {"ar": ar, "sim": round(s, 4), "trans": name_meta[ar]["trans"]}
            for ar, s in bottom
        ],
        "estimated_position": {
            "px":          round(est_px, 4),
            "py":          round(est_py, 4),
            "r":           round(est_r, 4),
            "level":       est_level,
            "level_label": level_labels[min(est_level, 2)],
        },
        # v2.1: dominant_wazn now aliases the QUERY term's own pattern
        # (backward-compatible key). Cluster pattern moved to its own field.
        "dominant_wazn":            query_wazn,
        "query_wazn":               query_wazn,
        "query_wazn_status":        query_wazn_status,
        "cluster_wazn":             cluster_wazn,
        "attractors_centered": [
            {"ar": ar, "sim_centered": round(s, 4),
             "trans": name_meta[ar]["trans"]}
            for ar, s in top_centered
        ],
        "repelled_centered": [
            {"ar": ar, "sim_centered": round(s, 4),
             "trans": name_meta[ar]["trans"]}
            for ar, s in bottom_centered
        ],
        "poincare_dist_to_primary": round(p_dist, 4),
    }

    # Save to dataset — dedup check inside save_to_dataset
    save_to_dataset({
        "term_ar":              term,
        "term_undiacritized":   strip_diac(term),
        "top_name_attractors":  [[ar, round(s, 4)] for ar, s in top],
        "bottom_names":         [[ar, round(s, 4)] for ar, s in bottom],
        "estimated_position":   result["estimated_position"],
        "dominant_wazn":        query_wazn,
        "query_wazn":           query_wazn,
        "query_wazn_status":    query_wazn_status,
        "cluster_wazn":         cluster_wazn,
        "poincare_dist_to_primary": round(p_dist, 4),
        "abjad_value":          term_abjad,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        # NOTE: embedding vector intentionally omitted
    })

    return result

# ── FastAPI app ──────────────────────────────────────────────────────────
import contextlib

app = FastAPI(
    title="AL-MIR'AH",
    servers=[{"url": "https://wellyes1-almiraah-transformer.hf.space"}],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"]
)

with open("ui.html", encoding="utf-8") as f:
    UI_HTML = f.read()

PRIVACY_HTML = ""
if os.path.exists("privacy.html"):
    with open("privacy.html", encoding="utf-8") as f:
        PRIVACY_HTML = f.read()

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=UI_HTML)

@app.get("/privacy", response_class=HTMLResponse)
@app.get("/privacy.html", response_class=HTMLResponse)
def privacy():
    return HTMLResponse(content=PRIVACY_HTML)

@app.get("/health")
def health():
    return {"status": "ok", "names_embedded": len(name_vecs),
            "dataset_terms": len(coord_db_terms)}

@app.get("/query")
def query(term: str):
    return JSONResponse(run_query(term))

@app.get("/neighbors")
def neighbors(term: str, k: int = 10, min_r: float = 0.1, max_r: float = 0.95):
    if not coord_db:
        return JSONResponse({"error": "coordinates.jsonl not loaded", "neighbors": []})

    term_clean = strip_diac(term)
    target = None
    for entry in coord_db:
        if entry.get("term_undiacritized") == term_clean or \
           strip_diac(entry.get("term_ar", "")) == term_clean:
            target = entry
            break

    if target is None:
        result = run_query(term)
        pos = result["estimated_position"]
        target_pos = (pos["px"], pos["py"])
        target_label = term
    else:
        pos = target["estimated_position"]
        target_pos = (pos["px"], pos["py"])
        target_label = target.get("term_ar", term)

    distances = []
    seen = set()
    excluded = 0
    for entry in coord_db:
        ar  = entry.get("term_ar", "")
        key = entry.get("term_undiacritized", strip_diac(ar))
        if key == term_clean or key in seen:
            continue
        seen.add(key)
        ep = entry.get("estimated_position", {})
        r  = ep.get("r", 0.0)
        if r < min_r or r > max_r:
            excluded += 1
            continue
        d = poincare_distance(target_pos, (ep.get("px", 0.0), ep.get("py", 0.0)))
        distances.append({
            "term_ar":            ar,
            "term_undiacritized": key,
            "distance":           round(d, 4),
            "r":                  round(r, 4),
            "level":              ep.get("level_label", ""),
            "top_attractors":     [a[0] for a in entry.get("top_name_attractors", [])[:3]],
        })

    distances.sort(key=lambda x: x["distance"])
    return JSONResponse({
        "term":           target_label,
        "position":       {"px": round(target_pos[0], 4), "py": round(target_pos[1], 4)},
        "k":              k,
        "filters":        {"min_r": min_r, "max_r": max_r},
        "total_searched": len(distances) + excluded,
        "excluded":       excluded,
        "neighbors":      distances[:k],
    })


# ── /project — cross-linguistic semantic projection ───────────────────
def get_full_profile(term_ar: str) -> dict:
    result = run_query(term_ar)
    pos    = result["estimated_position"]
    top    = result["top_names"]
    top1   = top[0] if top else {}
    axis_str = top1.get("paired_opposite", "") if top else ""

    attractors = [
        {
            "name_ar":         n["ar"],
            "name_trans":      n["trans"],
            "sim":             round(n["sim"], 4),
            "tier":            n.get("level", ""),
            "root":            n.get("root", ""),
            "abjad":           n.get("abjad", None),
            "wazn":            n.get("wazn", ""),
            "paired_opposite": n.get("paired_opposite", ""),
        }
        for n in top
    ]

    repelled = [
        {"name_ar": n["ar"], "name_trans": n["trans"], "sim": round(n["sim"], 4)}
        for n in result["bottom_names"]
    ]

    term_clean  = strip_diac(term_ar)
    target_pos  = (pos["px"], pos["py"])
    dists = []
    seen  = set()
    for entry in coord_db:
        ar  = entry.get("term_ar", "")
        key = entry.get("term_undiacritized", strip_diac(ar))
        if key == term_clean or key in seen:
            continue
        seen.add(key)
        ep = entry.get("estimated_position", {})
        r  = ep.get("r", 0.0)
        if r < 0.15 or r > 0.92:
            continue
        d = poincare_distance(target_pos, (ep.get("px", 0.0), ep.get("py", 0.0)))
        dists.append((ar, round(d, 4), round(r, 4), ep.get("level_label", ""),
                      [a[0] for a in entry.get("top_name_attractors", [])[:2]]))
    dists.sort(key=lambda x: x[1])

    return {
        "term_ar":           term_ar,
        "term_clean":        term_clean,
        "abjad":             result.get("abjad", {}),
        "position":          {"px": round(pos["px"], 4),
                              "py": round(pos["py"], 4),
                              "r":  round(pos["r"], 4)},
        "tier":              pos.get("level_label", ""),
        "dominant_wazn":     result.get("dominant_wazn", ""),
        "poincare_dist_to_primary": round(result.get("poincare_dist_to_primary", 0.0), 4),
        "axis":              axis_str,
        "attractors":        attractors,
        "repelled":          repelled,
        "dataset_neighbors": [
            {"term_ar": a, "distance": d, "r": r, "tier": lv, "top_attractors": at}
            for a, d, r, lv, at in dists[:5]
        ],
    }



def dual_distance(pos1: dict, pos2: dict) -> dict:
    """v3.1: the two measurements, side by side.

    distance_euclidean treats the disk as FLAT — displacement only,
    blind to depth. distance_hyperbolic is the geodesic the model
    actually lives in: the same displacement costs more metric the
    deeper into the rim (the more specific/differentiated) the pair
    sits. hierarchy_load = d_hyp / d_euc isolates that surcharge:
    ≈2.0–2.5 near the origin (flat regime, "linear" relations),
    growing without bound toward the rim (curved regime — relations
    mediated by hierarchical depth). Interpretive gloss, held lightly:
    low load = the two terms differ by simple translation in the
    field; high load = their difference is constituted by hierarchy
    and must be traversed through it.
    """
    import math
    dx = pos1["px"] - pos2["px"]; dy = pos1["py"] - pos2["py"]
    d_euc = math.sqrt(dx*dx + dy*dy)
    d_hyp = poincare_distance((pos1["px"], pos1["py"]),
                              (pos2["px"], pos2["py"]))
    load = round(d_hyp / d_euc, 3) if d_euc > 1e-6 else None
    return {"distance_euclidean": round(d_euc, 4),
            "distance_hyperbolic": round(d_hyp, 4),
            "hierarchy_load": load}


def context_centroid(profiles: list) -> tuple:
    if not profiles:
        return (0.0, 0.0)
    # v3.1: Karcher mean on the disk (was r-weighted Euclidean average)
    from hyperbolic import karcher_mean
    import numpy as _np
    pts = [_np.array([p["position"]["px"], p["position"]["py"]]) for p in profiles]
    ws  = [max(p["position"]["r"], 1e-3) for p in profiles]
    km = karcher_mean(pts, ws)
    return (float(km[0]), float(km[1]))


def geometric_fit(profile: dict, centroid: tuple) -> float:
    d = poincare_distance(
        (profile["position"]["px"], profile["position"]["py"]),
        centroid
    )
    return round(1.0 / (1.0 + d), 4)


@app.get("/compare")
def compare(term1: str, term2: str):
    r1 = run_query(term1)
    r2 = run_query(term2)
    top1 = {n["ar"]: n for n in r1["top_names"]}
    top2 = {n["ar"]: n for n in r2["top_names"]}
    shared   = set(top1) & set(top2)
    only_t1  = set(top1) - set(top2)
    only_t2  = set(top2) - set(top1)
    bot1 = {n["ar"] for n in r1["bottom_names"]}
    bot2 = {n["ar"] for n in r2["bottom_names"]}
    opposing = (set(top1) & bot2) | (set(top2) & bot1)
    pos1 = r1["estimated_position"]
    pos2 = r2["estimated_position"]
    dd = dual_distance(pos1, pos2)
    return JSONResponse({
        "term1": term1, "term2": term2,
        "distance_euclidean":  dd["distance_euclidean"],
        "distance_hyperbolic": dd["distance_hyperbolic"],
        "hierarchy_load":      dd["hierarchy_load"],
        "term1_position": {"r": pos1["r"], "level": pos1["level_label"],
                           "dominant_wazn": r1["dominant_wazn"],
                           "abjad": r1.get("abjad", {})},
        "term2_position": {"r": pos2["r"], "level": pos2["level_label"],
                           "dominant_wazn": r2["dominant_wazn"],
                           "abjad": r2.get("abjad", {})},
        "shared_attractors": [
            {"ar": ar, "trans": top1[ar]["trans"],
             "sim_t1": round(top1[ar]["sim"], 4),
             "sim_t2": round(top2[ar]["sim"], 4)}
            for ar in shared
        ],
        "divergent_t1_only": [
            {"ar": ar, "trans": top1[ar]["trans"], "sim": round(top1[ar]["sim"], 4)}
            for ar in only_t1
        ],
        "divergent_t2_only": [
            {"ar": ar, "trans": top2[ar]["trans"], "sim": round(top2[ar]["sim"], 4)}
            for ar in only_t2
        ],
        "opposing_poles": list(opposing),
    })


@app.post("/project")
async def project(request: Request):
    body = await request.json()
    context_arabic = body.get("context_arabic", [])
    candidates_map = body.get("candidates", {})
    if not candidates_map:
        return JSONResponse({"error": "candidates field required"}, status_code=400)

    context_profiles = []
    for term in context_arabic:
        if term.strip():
            try:
                context_profiles.append(get_full_profile(term.strip()))
            except Exception as e:
                log.warning(f"Context term failed: {term} — {e}")

    centroid   = context_centroid(context_profiles)
    centroid_r = round(float(np.sqrt(centroid[0]**2 + centroid[1]**2)), 4)
    results    = {}

    for concept, arabic_forms in candidates_map.items():
        candidate_profiles = []
        for form in arabic_forms:
            if not form.strip():
                continue
            try:
                profile = get_full_profile(form.strip())
                profile["fit_score"] = (
                    geometric_fit(profile, centroid) if context_profiles else None
                )
                candidate_profiles.append(profile)
            except Exception as e:
                log.warning(f"Candidate failed: {form} — {e}")

        if context_profiles and candidate_profiles:
            candidate_profiles.sort(
                key=lambda p: p["fit_score"] if p["fit_score"] is not None else 0,
                reverse=True
            )
            best = candidate_profiles[0]
            justification = (
                f"{best['term_ar']} scores highest (fit={best['fit_score']}) — "
                f"tier: {best['tier']}, dominant wazn: {best['dominant_wazn']}, "
                f"primary axis: {best['attractors'][0]['name_ar'] if best['attractors'] else '—'} "
                f"↔ {best['attractors'][0]['paired_opposite'][:40] if best['attractors'] else '—'}. "
                f"Context centroid at ({round(centroid[0],3)}, {round(centroid[1],3)}) r={centroid_r}."
            )
        else:
            best = candidate_profiles[0] if candidate_profiles else None
            justification = "No context terms provided — fit scores unavailable."

        results[concept] = {
            "candidates":    candidate_profiles,
            "recommended":   best["term_ar"] if best else None,
            "justification": justification,
        }

    return JSONResponse({
        "context_centroid": {"px": round(centroid[0], 4),
                             "py": round(centroid[1], 4), "r": centroid_r},
        "context_profiles": context_profiles,
        "projections":      results,
    })


# ── MCP Server ────────────────────────────────────────────────────────
mcp_server = Server("almiraah-philological")

@mcp_server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="philological_lookup",
            description=(
                "Look up the philological coordinate of an Arabic term in the "
                "99-Name field. Returns: Abjad value with per-letter breakdown, "
                "top attractor Names (tier, similarity, semantic description, "
                "root, Abjad value, wazn, axis/paired-opposite), structurally "
                "absent/repelled Names, estimated Poincaré disk position (px, py, r), "
                "hierarchy level (Dhāt/Ṣifāt/Afʿāl), and dominant morphological "
                "pattern. Use this to ground any semantic claim about an Arabic "
                "term in philological coordinate data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "Arabic term to look up (diacritics optional)",
                    }
                },
                "required": ["term"],
            },
        ),
        types.Tool(
            name="root_analysis",
            description=(
                "Analyse all 99 Names sharing a given Arabic root (3 consonants). "
                "Returns each matching Name with: Abjad value, wazn, hierarchy level, "
                "paired opposite (axis partner), ML homolog, and available phonetic, "
                "numerical, geometric, and breath-layer annotations. "
                "Use to explore the full semantic field of a root across the divine Names."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": "Three Arabic consonants separated by hyphens, e.g. ر-ح-م",
                    }
                },
                "required": ["root"],
            },
        ),
        types.Tool(
            name="semantic_project",
            description=(
                "Cross-linguistic semantic projection. Given candidate Arabic forms "
                "for a concept and Arabic context terms, projects each candidate "
                "through the 99-Names field and returns: tier, axis, wazn, Abjad "
                "breakdown, attractor Names, repelled Names, dataset neighbors, "
                "and geometric fit score against the context centroid. "
                "Use to select the geometrically correct Arabic form for a context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "object",
                        "description": "Map of concept → list of Arabic candidate forms. E.g. {'dissolution': ['فناء','زوال','محو']}",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "context_arabic": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arabic context terms — used to compute the semantic centroid.",
                    },
                },
                "required": ["candidates"],
            },
        ),
        types.Tool(
            name="semantic_neighbors",
            description=(
                "Find Arabic terms in the accumulated query dataset geometrically "
                "closest to a given term by Poincaré distance. Returns terms ranked "
                "by proximity in the 99-Names field with tier, r-value, and top "
                "attractor Names. Supports min_r/max_r quality filters."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "term":  {"type": "string", "description": "Arabic term"},
                    "k":     {"type": "integer", "description": "Neighbors to return (default 8)", "default": 8},
                    "min_r": {"type": "number",  "description": "Minimum r filter (default 0.1)", "default": 0.1},
                    "max_r": {"type": "number",  "description": "Maximum r filter (default 0.95)", "default": 0.95},
                },
                "required": ["term"],
            },
        ),
        types.Tool(
            name="compare_terms",
            description=(
                "Compare two Arabic terms in the 99-Name field. Returns: shared "
                "attractor Names, divergent Names unique to each term, opposing "
                "poles (one attracts what the other repels), coordinate distance, "
                "hierarchy levels, and Abjad values for both terms. "
                "Use to determine structural relationship between two concepts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "term1": {"type": "string", "description": "First Arabic term"},
                    "term2": {"type": "string", "description": "Second Arabic term"},
                },
                "required": ["term1", "term2"],
            },
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    if name == "philological_lookup":
        term   = arguments["term"]
        result = run_query(term)
        pos    = result["estimated_position"]
        abjad  = result.get("abjad", {})

        top_str = "\n".join(
            f"  {i+1}. {n['ar']} ({n['trans']}) — sim: {n['sim']}\n"
            f"     tier: {['Dhāt','Ṣifāt','Afʿāl'][min(n['level'],2)]} | "
            f"root: {n['root']} | abjad: {n['abjad']} | wazn: {n['wazn']}\n"
            f"     axis: {n['ar']} ⇄ {n['paired_opposite']}\n"
            f"     meaning: {n['meaning']}\n"
            f"     {n['layer2_semantic']}"
            for i, n in enumerate(result["top_names"])
        )
        bot_str = ", ".join(
            f"{n['ar']} ({n['trans']}, sim={n['sim']})"
            for n in result["bottom_names"]
        )
        cen_str = ", ".join(
            f"{c['ar']} ({c['sim_centered']:+.3f})"
            for c in result.get("attractors_centered", [])
        )
        cen_rep_str = ", ".join(
            f"{c['ar']} ({c['sim_centered']:+.3f})"
            for c in result.get("repelled_centered", [])
        )
        output = (
            f"PHILOLOGICAL COORDINATE: {term}\n"
            f"{'─'*52}\n"
            f"Abjad value: {abjad.get('value','—')}  [{abjad.get('breakdown','—')}]\n"
            f"Position: px={pos['px']}, py={pos['py']}, r={pos['r']}\n"
            f"Hierarchy: {pos['level_label']} (level {pos['level']})\n"
            f"Query wazn: {result['query_wazn']}  [{result['query_wazn_status']}]\n"
            f"Cluster wazn (attractor majority): {result['cluster_wazn']}\n"
            f"Poincaré dist to primary: {result['poincare_dist_to_primary']}\n\n"
            f"TOP ATTRACTOR NAMES (raw):\n{top_str}\n\n"
            f"IʿTIDĀL-CENTERED ATTRACTORS (deviation from field zero μ — {FIELD_ZERO_SOURCE['value']}):\n"
            f"  {cen_str}\n"
            f"IʿTIDĀL-CENTERED REPELLED:\n  {cen_rep_str}\n\n"
            f"STRUCTURALLY ABSENT (repelled, raw):\n  {bot_str}\n"
        )
        return [types.TextContent(type="text", text=output)]

    elif name == "root_analysis":
        root    = arguments["root"]
        matches = [(ar, meta) for ar, meta in name_meta.items()
                   if meta["root"] == root]
        if not matches:
            return [types.TextContent(type="text",
                text=f"No Names found with root {root}. "
                     f"Try format ر-ح-م (consonants separated by hyphens).")]
        lines = [f"ROOT ANALYSIS: {root}", "─"*44]
        for ar, meta in matches:
            tier = ['Dhāt','Ṣifāt','Afʿāl'][min(meta['level'], 2)]
            lines.append(f"\n{ar} ({meta['trans']})")
            _af = re.sub(r'^ال', '', strip_diac(ar))
            _recomp = sum(ABJAD.get(c, 0) for c in _af)
            _flag = "" if _recomp == meta['abjad'] else "  ⚠ stored≠recomputed (pending Abjad audit)"
            lines.append(f"  Abjad (stored): {meta['abjad']} | recomputed article-free: {abjad_breakdown(_af)}{_flag}")
            lines.append(f"  Wazn: {meta['wazn']} | Tier: {tier}")
            lines.append(f"  Meaning: {meta['meaning']}")
            lines.append(f"  Axis: {ar} ⇄ {meta['paired_opposite'][:80]}")
            if meta.get("ml_homolog"):
                lines.append(f"  ML homolog: {meta['ml_homolog'][:100]}")
            if meta.get("layer1_phonetic"):
                lines.append(f"  Phonetic: {meta['layer1_phonetic'][:100]}")
            if meta.get("layer3_numerical"):
                lines.append(f"  Numerical: {meta['layer3_numerical'][:100]}")
            if meta.get("layer4_geometric"):
                lines.append(f"  Geometric: {meta['layer4_geometric'][:100]}")
            if meta.get("layer5_breath"):
                lines.append(f"  Breath: {meta['layer5_breath'][:100]}")
        return [types.TextContent(type="text", text="\n".join(lines))]

    elif name == "semantic_project":
        candidates_map = arguments.get("candidates", {})
        context_arabic = arguments.get("context_arabic", [])

        if not candidates_map:
            return [types.TextContent(type="text", text="Error: candidates required")]

        context_profiles = []
        for term in context_arabic:
            if term.strip():
                try:
                    context_profiles.append(get_full_profile(term.strip()))
                except Exception:
                    pass

        centroid   = context_centroid(context_profiles)
        centroid_r = round(float(np.sqrt(centroid[0]**2 + centroid[1]**2)), 4)
        divider    = "─" * 44
        lines      = ["SEMANTIC PROJECTION", divider]

        if context_arabic:
            lines.append(f"Context: {', '.join(context_arabic)}")
            lines.append(f"Centroid: ({round(centroid[0],3)}, {round(centroid[1],3)}) r={centroid_r}")
            lines.append(divider)

        for concept, arabic_forms in candidates_map.items():
            lines.append(f"CONCEPT: {concept}")
            profiles = []
            for form in arabic_forms:
                if not form.strip():
                    continue
                try:
                    p = get_full_profile(form.strip())
                    p["fit_score"] = geometric_fit(p, centroid) if context_profiles else None
                    profiles.append(p)
                except Exception as e:
                    lines.append(f"  {form}: error — {e}")

            if context_profiles:
                profiles.sort(
                    key=lambda p: p["fit_score"] if p["fit_score"] is not None else 0,
                    reverse=True
                )

            for p in profiles:
                fit_str  = f"fit={p['fit_score']}" if p["fit_score"] is not None else "no context"
                top_attr = p["attractors"][0] if p["attractors"] else {}
                abjad_v  = p.get("abjad", {}).get("value", "—")
                lines.append(
                    f"  {p['term_ar']} | {fit_str} | r={p['position']['r']} "
                    f"| {p['tier']} | wazn:{p['dominant_wazn']} | abjad:{abjad_v}"
                )
                lines.append(
                    f"    axis: {top_attr.get('name_ar','—')} ⇄ "
                    f"{(top_attr.get('paired_opposite','') or '')[:40]}"
                )
                if p["dataset_neighbors"]:
                    nbr_str = ", ".join(n["term_ar"] for n in p["dataset_neighbors"][:3])
                    lines.append(f"    neighbors: {nbr_str}")

            if profiles:
                best = profiles[0]
                fit_str = f"fit={best['fit_score']}" if best["fit_score"] is not None else "highest r"
                lines.append(f"  → RECOMMENDED: {best['term_ar']} ({fit_str})")
            lines.append("")

        return [types.TextContent(type="text", text="\n".join(lines))]

    elif name == "semantic_neighbors":
        term  = arguments.get("term", "")
        k     = int(arguments.get("k", 8))
        min_r = float(arguments.get("min_r", 0.1))
        max_r = float(arguments.get("max_r", 0.95))
        if not term:
            return [types.TextContent(type="text", text="Error: term is required")]
        if not coord_db:
            return [types.TextContent(type="text",
                text="Dataset not loaded — neighbors unavailable")]

        term_clean = strip_diac(term)
        target = None
        for entry in coord_db:
            if entry.get("term_undiacritized") == term_clean or \
               strip_diac(entry.get("term_ar", "")) == term_clean:
                target = entry
                break

        if target is None:
            result     = run_query(term)
            pos        = result["estimated_position"]
            target_pos = (pos["px"], pos["py"])
            target_label = term
            source     = "computed on-the-fly"
        else:
            ep         = target["estimated_position"]
            target_pos = (ep["px"], ep["py"])
            target_label = target.get("term_ar", term)
            source     = "found in dataset"

        distances = []
        seen = set()
        for entry in coord_db:
            ar  = entry.get("term_ar", "")
            key = entry.get("term_undiacritized", strip_diac(ar))
            if key == term_clean or key in seen:
                continue
            seen.add(key)
            ep = entry.get("estimated_position", {})
            r  = ep.get("r", 0.0)
            if r < min_r or r > max_r:
                continue
            d = poincare_distance(target_pos, (ep.get("px", 0.0), ep.get("py", 0.0)))
            distances.append((ar, round(d, 4), round(r, 4),
                               ep.get("level_label", ""),
                               [a[0] for a in entry.get("top_name_attractors", [])[:3]]))

        distances.sort(key=lambda x: x[1])
        divider = "─" * 40
        lines = [
            f"SEMANTIC NEIGHBORS: {target_label}",
            f"Position: ({round(target_pos[0],4)}, {round(target_pos[1],4)}) | {source}",
            f"Filters: min_r={min_r}, max_r={max_r} | Top {k}",
            divider,
        ]
        for i, (ar, dist, r, level, attrs) in enumerate(distances[:k], 1):
            attr_str = ", ".join(attrs) if attrs else "—"
            lines.append(f"{i}. {ar} | dist={dist} | r={r} | {level}")
            lines.append(f"   attractors: {attr_str}")

        return [types.TextContent(type="text", text="\n".join(lines))]

    elif name == "compare_terms":
        t1, t2 = arguments["term1"], arguments["term2"]
        r1 = run_query(t1)
        r2 = run_query(t2)

        top1 = {n["ar"]: n for n in r1["top_names"]}
        top2 = {n["ar"]: n for n in r2["top_names"]}
        shared   = set(top1) & set(top2)
        only_t1  = set(top1) - set(top2)
        only_t2  = set(top2) - set(top1)
        bot1     = {n["ar"] for n in r1["bottom_names"]}
        bot2     = {n["ar"] for n in r2["bottom_names"]}
        opposing = (set(top1) & bot2) | (set(top2) & bot1)

        pos1 = r1["estimated_position"]
        pos2 = r2["estimated_position"]
        dd = dual_distance(pos1, pos2)

        a1 = r1.get("abjad", {}); a2 = r2.get("abjad", {})
        shared_str   = ", ".join(f"{ar} ({top1[ar]['trans']})" for ar in shared)   or "none"
        only_t1_str  = ", ".join(f"{ar} ({top1[ar]['trans']})" for ar in only_t1)  or "none"
        only_t2_str  = ", ".join(f"{ar} ({top2[ar]['trans']})" for ar in only_t2)  or "none"
        opposing_str = ", ".join(opposing) or "none"

        output = (
            f"COMPARISON: {t1}  ↔  {t2}\n"
            f"{'─'*52}\n"
            f"{t1}: r={pos1['r']}, {pos1['level_label']}, wazn={r1['dominant_wazn']}, "
            f"abjad={a1.get('value','—')} [{a1.get('breakdown','—')}]\n"
            f"{t2}: r={pos2['r']}, {pos2['level_label']}, wazn={r2['dominant_wazn']}, "
            f"abjad={a2.get('value','—')} [{a2.get('breakdown','—')}]\n"
            f"Distance — Euclidean (flat): {dd['distance_euclidean']} | hyperbolic (geodesic): {dd['distance_hyperbolic']} | hierarchy load: {dd['hierarchy_load']}\n\n"
            f"SHARED ATTRACTORS: {shared_str}\n\n"
            f"DIVERGENT — {t1} only: {only_t1_str}\n"
            f"DIVERGENT — {t2} only: {only_t2_str}\n\n"
            f"OPPOSING POLES (one attracts, other repels): {opposing_str}\n"
        )
        return [types.TextContent(type="text", text=output)]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ── MCP Session Manager (must be after mcp_server is defined) ─────────────
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

session_manager = StreamableHTTPSessionManager(
    app=mcp_server,
    json_response=False,
    stateless=False,
)

@contextlib.asynccontextmanager
async def lifespan(app):
    async with session_manager.run():
        yield

app.router.lifespan_context = lifespan


# ── OAuth metadata (required for Claude.ai remote MCP connector) ──────
from fastapi.responses import RedirectResponse

BASE_URL = "https://wellyes1-almiraah-transformer.hf.space"

@app.get("/.well-known/oauth-authorization-server")
async def oauth_metadata():
    return JSONResponse({
        "issuer": BASE_URL,
        "authorization_endpoint": f"{BASE_URL}/oauth/authorize",
        "token_endpoint":         f"{BASE_URL}/oauth/token",
        "registration_endpoint":  f"{BASE_URL}/register",
        "scopes_supported":              ["mcp"],
        "response_types_supported":      ["code"],
        "grant_types_supported":         ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
    })

@app.get("/oauth/authorize")
async def oauth_authorize(
    redirect_uri: str = "", state: str = "",
    response_type: str = "code", client_id: str = "",
    code_challenge: str = "", code_challenge_method: str = "S256",
):
    return RedirectResponse(
        url=f"{redirect_uri}?code=almiraah-open-access&state={state}"
    )

@app.post("/oauth/token")
async def oauth_token(request: Request):
    return JSONResponse({
        "access_token": "almiraah-open-access-token",
        "token_type":   "bearer",
        "expires_in":   86400,
        "scope":        "mcp",
    })

@app.post("/register")
async def oauth_register(request: Request):
    body = await request.json()
    return JSONResponse({
        "client_id":          "almiraah-public-client",
        "client_secret":      "almiraah-open-access",
        "client_id_issued_at": 0,
        "grant_types":        ["authorization_code"],
        "redirect_uris":      body.get("redirect_uris", []),
    }, status_code=201)


# ── MCP Streamable HTTP endpoint ─────────────────────────────────────
# Session manager handles all session state across requests.
# Claude.ai MCP URL: https://wellyes1-almiraah-transformer.hf.space/mcp

@app.get("/mcp")
@app.post("/mcp")
@app.delete("/mcp")
async def mcp_endpoint(request: Request):
    """Streamable HTTP MCP endpoint — all methods routed through session manager."""
    await session_manager.handle_request(
        request.scope, request.receive, request._send
    )

# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)