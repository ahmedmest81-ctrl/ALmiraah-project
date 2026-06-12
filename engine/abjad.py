# -*- coding: utf-8 -*-
"""
abjad.py — Single canonical Abjad computation for AL-MIR'ĀH.

BUG (live-confirmed 2026-06-10, root_analysis ق-د-ر):
    القَادِر header said "Abjad: 305" while its per-letter breakdown
    summed 336 — the breakdown included ال (ا=1 + ل=30 = 31), the
    header did not. المُقْتَدِر: header 745 vs breakdown 775. Same 31.

POLICY (one convention, stated everywhere):
    The canonical Abjad value of a Name or term is computed WITHOUT
    the definite article — the article is grammar, not lexicon, and
    the all_99_corrected.json `abjad_num` values follow this
    convention (e.g. القَادِر → 305 over قادر). The article-inclusive
    value may be reported as a separate, explicitly labeled field.
    Header and breakdown must be produced by THE SAME call so they
    can never diverge again.
"""

import re
import unicodedata

MASHRIQI = {
    'ا': 1, 'ب': 2, 'ج': 3, 'د': 4, 'ه': 5, 'و': 6, 'ز': 7, 'ح': 8,
    'ط': 9, 'ي': 10, 'ك': 20, 'ل': 30, 'م': 40, 'ن': 50, 'س': 60,
    'ع': 70, 'ف': 80, 'ص': 90, 'ق': 100, 'ر': 200, 'ش': 300, 'ت': 400,
    'ث': 500, 'خ': 600, 'ذ': 700, 'ض': 800, 'ظ': 900, 'غ': 1000,
    # orthographic variants
    'أ': 1, 'إ': 1, 'آ': 1, 'ء': 1, 'ؤ': 6, 'ئ': 10,
    'ة': 400,   # tāʾ marbūṭah valued as tāʾ (project convention; see
                # dataset_field_audit.md — keep consistent with Paper A)
    'ى': 10,    # alif maqṣūrah as yāʾ
}

_DIACRITICS = re.compile(r'[\u064B-\u0652\u0670\u0640]')
_ARTICLE = re.compile(r'^ال')


def abjad(term: str, include_article: bool = False) -> dict:
    """
    Returns {'value', 'breakdown', 'form_used', 'convention'} where
    breakdown is the list of (letter, value) pairs whose sum IS the
    value — by construction, not by separate computation.
    """
    t = unicodedata.normalize('NFC', term.strip())
    t = _DIACRITICS.sub('', t)
    form = t if include_article else _ARTICLE.sub('', t)
    breakdown = [(ch, MASHRIQI[ch]) for ch in form if ch in MASHRIQI]
    return {
        'value': sum(v for _, v in breakdown),
        'breakdown': breakdown,
        'form_used': form,
        'convention': ('article-inclusive' if include_article
                       else 'article-free (canonical)'),
    }


def format_line(term: str) -> str:
    """Server-side formatter — header and breakdown from one call."""
    a = abjad(term)
    parts = ' + '.join(f'{l}={v}' for l, v in a['breakdown'])
    return f"Abjad: {a['value']}  [{parts} = {a['value']}]"


if __name__ == '__main__':
    import json
    names = json.load(open('/mnt/project/all_99_corrected.json'))
    mismatches = []
    for n in names:
        computed = abjad(n['name_ar'])['value']
        stored = n.get('abjad_num')
        if stored is not None and computed != stored:
            mismatches.append((n['name_ar'], stored, computed))
    print(f'checked 99 Names against stored abjad_num: '
          f'{len(mismatches)} mismatches')
    for ar, st, co in mismatches[:15]:
        print(f'  {ar}: stored={st} computed(article-free)={co} '
              f'computed(with article)={abjad(ar, True)["value"]}')
