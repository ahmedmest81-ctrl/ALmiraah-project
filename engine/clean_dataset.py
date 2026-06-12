# -*- coding: utf-8 -*-
"""
clean_dataset.py — Hygiene pass for the accumulated query dataset
(production file: 1779798638098_coordinates__4_.jsonl, 722 records).

Live-confirmed contamination (2026-06-10 tool test):
  1. Mojibake: at least one record stores double-encoded UTF-8
     ("Ø±Ø­Ù…Ø©" = رحمة encoded UTF-8 → decoded latin-1 → re-encoded).
  2. Calibration noise: number-words (عشرون، مائتان، ...) from early
     calibration queries pollute semantic_neighbors results.
  3. Possible duplicates across query sessions.

Policy: NON-DESTRUCTIVE. Writes <name>.cleaned.jsonl plus a
<name>.removed.jsonl audit file. Nothing is silently dropped —
every removal carries a reason. The cleaned file becomes the
reference set for computing the field zero μ (equilibrium.py).
"""

import json
import re
import sys
import unicodedata
from collections import OrderedDict

# ----------------------------------------------------------------------
# 1. Mojibake repair
# ----------------------------------------------------------------------

_MOJIBAKE_MARKERS = set('ÃÂØÙÐ¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿')


def looks_mojibake(s: str) -> bool:
    return any(ch in _MOJIBAKE_MARKERS for ch in s)


def repair_mojibake(s: str) -> tuple[str, bool]:
    """UTF-8 read as latin-1/cp1252 → round-trip back. Repairs up to
    two layers of double-encoding; returns (repaired, changed)."""
    cur, changed = s, False
    for _ in range(2):
        if not looks_mojibake(cur):
            break
        for enc in ('latin-1', 'cp1252'):
            try:
                cand = cur.encode(enc).decode('utf-8')
                if not looks_mojibake(cand) or len(cand) < len(cur):
                    cur, changed = cand, True
                    break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        else:
            break
    return cur, changed


# ----------------------------------------------------------------------
# 2. Calibration-noise detection
# ----------------------------------------------------------------------

NUMBER_WORDS = {
    'واحد', 'اثنان', 'اثنين', 'ثلاثة', 'أربعة', 'خمسة', 'ستة', 'سبعة',
    'ثمانية', 'تسعة', 'عشرة', 'عشرون', 'ثلاثون', 'أربعون', 'خمسون',
    'ستون', 'سبعون', 'ثمانون', 'تسعون', 'مائة', 'مائتان', 'مئة', 'مئتان',
    'ألف', 'ألفان', 'مليون',
}
# NOTE deliberately NOT excluded: أحد، صفر — أحد collides with the Name
# Al-Aḥad's lexical field and صفر (zero) is theoretically loaded for this
# project. Review such cases manually rather than auto-dropping.

_LATIN = re.compile(r'[A-Za-z]')
_DIGITS = re.compile(r'[0-9\u0660-\u0669]')


def noise_reason(term: str) -> str | None:
    t = unicodedata.normalize('NFC', term.strip())
    bare = re.sub(r'^(ال)', '', t)
    if bare in NUMBER_WORDS or t in NUMBER_WORDS:
        return 'calibration_number_word'
    if _DIGITS.search(t):
        return 'contains_digits'
    if _LATIN.search(t) and not re.search(r'[\u0600-\u06FF]', t):
        return 'non_arabic'
    if len(re.sub(r'[^\u0600-\u06FF]', '', t)) < 2:
        return 'too_short'
    return None


# ----------------------------------------------------------------------
# 3. Validation + dedup
# ----------------------------------------------------------------------

def validate(rec: dict) -> str | None:
    r = rec.get('r')
    if r is not None and not (0.0 <= float(r) <= 1.0):
        return f'r_out_of_range:{r}'
    px, py = rec.get('px'), rec.get('py')
    if px is not None and py is not None:
        if (float(px) ** 2 + float(py) ** 2) ** 0.5 > 1.0 + 1e-9:
            return 'outside_poincare_disk'
    return None


def term_key(rec: dict) -> str:
    for k in ('term_ar', 'term', 'query', 'name_ar', 'word'):
        if k in rec and rec[k]:
            return unicodedata.normalize('NFC', str(rec[k]).strip())
    return json.dumps(rec, ensure_ascii=False)[:64]


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def clean(path_in: str):
    kept, removed, repaired_count = OrderedDict(), [], 0
    with open(path_in, encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                removed.append({'line': ln, 'reason': 'json_error', 'raw': line[:200]})
                continue

            # mojibake repair on every string field, recursively
            def fix(obj):
                nonlocal repaired_count
                if isinstance(obj, str):
                    new, ch = repair_mojibake(obj)
                    repaired_count += ch
                    return new
                if isinstance(obj, list):
                    return [fix(x) for x in obj]
                if isinstance(obj, dict):
                    return {k: fix(v) for k, v in obj.items()}
                return obj
            rec = fix(rec)

            t = term_key(rec)
            reason = noise_reason(t) or validate(rec)
            if reason:
                removed.append({'line': ln, 'term': t, 'reason': reason})
                continue
            if t in kept:
                removed.append({'line': ln, 'term': t, 'reason': 'duplicate'})
                continue
            kept[t] = rec

    out, audit = path_in + '.cleaned.jsonl', path_in + '.removed.jsonl'
    with open(out, 'w', encoding='utf-8') as f:
        for rec in kept.values():
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    with open(audit, 'w', encoding='utf-8') as f:
        for r in removed:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'kept={len(kept)}  removed={len(removed)}  '
          f'mojibake_repaired_fields={repaired_count}')
    print(f'→ {out}\n→ {audit}')
    reasons = {}
    for r in removed:
        reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f'   {k}: {v}')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        clean(sys.argv[1])
    else:
        # self-test on synthetic contaminated data
        import tempfile, os
        rows = [
            {'term': 'مرآة', 'r': 0.59, 'px': 0.37, 'py': -0.45},
            {'term': 'عشرون', 'r': 0.63, 'px': 0.1, 'py': 0.1},
            {'term': 'Ø±Ø­Ù…Ø©', 'r': 0.4, 'px': 0.0, 'py': 0.2},
            {'term': 'مرآة', 'r': 0.59, 'px': 0.37, 'py': -0.45},
            {'term': 'سر', 'r': 1.7, 'px': 0.9, 'py': 0.9},
        ]
        p = tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False,
                                        encoding='utf-8')
        for r in rows:
            p.write(json.dumps(r, ensure_ascii=False) + '\n')
        p.close()
        clean(p.name)
        print('\ncleaned content:')
        print(open(p.name + '.cleaned.jsonl', encoding='utf-8').read())
        os.unlink(p.name)
