# -*- coding: utf-8 -*-
"""
wazn.py — Standalone morphological-pattern (wazn) parser for AL-MIR'ĀH.

Fixes the inheritance bug: `dominant_wazn` previously reported the attractor
cluster's pattern instead of parsing the queried term. This module parses the
QUERY TERM ITSELF by positional radical substitution against a curated
template inventory (the classical miqyās method, ف-ع-ل as measure).

Design principle (الوقوف عند حدّ ما يُعلم):
Unvocalized Arabic underdetermines the wazn. Where the surface string admits
multiple patterns, this parser RETURNS ALL CANDIDATES with status
'ambiguous_vocalization' rather than guessing. The confidence gate is in the
architecture, not in discretion.

Output contract (three separate fields — never conflated):
    query_wazn        : str  — best single pattern, or slash-joined candidates
    query_wazn_status : str  — 'parsed' | 'ambiguous_vocalization' |
                               'ambiguous_structure' | 'not_parsed'
    candidates        : list — every template consistent with the surface form
The caller (server) reports cluster wazn separately as `cluster_wazn`.
"""

import re
import unicodedata

# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------

_DIACRITICS = re.compile(r'[\u064B-\u0652\u0670\u0640]')  # ḥarakāt, dagger alif, tatweel
_ARTICLE = re.compile(r'^(ال|أل|اَل)')

HAMZA_FORMS = {'أ': 'ء', 'إ': 'ء', 'ؤ': 'ء', 'ئ': 'ء'}
# madda = alif + hamza fused; expand so the radical hamza is recoverable
MADDA_EXPAND = {'آ': 'ءا'}


def normalize(term: str, strip_article: bool = True) -> str:
    t = unicodedata.normalize('NFC', term.strip())
    t = _DIACRITICS.sub('', t)
    if strip_article:
        t = _ARTICLE.sub('', t)
    for k, v in MADDA_EXPAND.items():
        t = t.replace(k, v)
    return t


# ----------------------------------------------------------------------
# Template inventory
# ----------------------------------------------------------------------
# Templates are written over the surface skeleton of an UNVOCALIZED string.
# Symbols: C = radical consonant slot; literal letters are augments
# (ا و ي م ت ن س ة ء in their structural positions).
# Each entry: (surface_regex_over_slots, wazn_name, n_radicals)
#
# The regex is built positionally: 'C' matches one consonant; everything
# else matches literally. Weak radicals (و ي ء) may also FILL a C slot —
# handled by making C match any Arabic letter, then checking augment
# positions are exactly the literals.

TEMPLATES = [
    # ---- triliteral, no augments (unvocalized CCC) -------------------
    ('CCC',    ['faʿl', 'fiʿl', 'fuʿl', 'faʿal', 'faʿil', 'faʿul'], 3),
    # ---- long-vowel-marked triliterals -------------------------------
    ('CاCC',   ['fāʿil'], 3),            # كاتب — unvocalized fāʿil
    ('CCيC',   ['faʿīl'], 3),            # كريم
    ('CCوC',   ['faʿūl', 'fuʿūl'], 3),   # صبور / سكوت — vocalization decides
    ('CCاC',   ['fiʿāl', 'faʿāl', 'fuʿāl'], 3),  # كتاب / سلام
    ('CCCان',  ['fiʿlān', 'faʿlān', 'fuʿlān'], 3),  # نسيان / رحمان
    ('CCاCة',  ['fiʿālah', 'faʿālah'], 3),
    ('CCيCة',  ['faʿīlah'], 3),          # سكينة
    ('CCCة',   ['faʿlah', 'fiʿlah', 'fuʿlah'], 3),  # عتبة
    ('CاCCة',  ['fāʿilah'], 3),
    # ---- mīm-prefixed ------------------------------------------------
    ('مCCC',   ['mafʿal', 'mifʿal', 'mufʿil', 'mufʿal'], 3),
    ('مCCCة',  ['mafʿalah', 'mifʿalah', 'mufʿilah'], 3),   # مرآة (with ء as radical)
    ('مCCوC',  ['mafʿūl'], 3),           # مكتوب
    ('مCاCC',  ['mufāʿil', 'mafāʿil'], 3),
    ('مCCّC',  ['mufaʿʿil', 'mufaʿʿal'], 3),   # rarely visible unvocalized (shadda stripped)
    ('مCتCC',  ['muftaʿil', 'muftaʿal'], 3),   # مقتدر
    ('منCCC',  ['munfaʿil'], 3),
    ('مستCCC', ['mustafʿil', 'mustafʿal'], 3),
    # ---- t-augmented -------------------------------------------------
    ('تCCيC',  ['tafʿīl'], 3),           # توحيد? careful: و is radical there → CCCيC fallback below
    ('تCاCC',  ['tafāʿul'], 3),
    ('CتCاC',  ['iftiʿāl-stem'], 3),
    ('اCتCاC', ['iftiʿāl'], 3),          # اقتدار
    ('انCCاC', ['infiʿāl'], 3),
    ('استCCاC', ['istifʿāl'], 3),
    ('اCCاC',  ['ifʿāl', 'afʿāl'], 3),   # إدراك (إ normalized to ء... see note)
    ('ءCCاC',  ['ifʿāl'], 3),
    # ---- doubled second radical (shadda invisible unvocalized) -------
    ('CCاC_intensive', ['faʿʿāl'], 3),   # handled as annotation on CCاC
    # ---- quadriliteral fallback --------------------------------------
    ('CCCC',   ['faʿlal (quadriliteral) or unvocalized-ambiguous'], 4),
]

_AR_LETTER = r'[\u0621-\u064A]'


def _template_to_regex(skel: str) -> str:
    out = []
    for ch in skel:
        if ch == 'C':
            out.append(f'({_AR_LETTER})')
        elif ch == 'ّ':
            continue  # shadda stripped during normalization
        else:
            out.append(re.escape(ch))
    return '^' + ''.join(out) + '$'


_COMPILED = [(re.compile(_template_to_regex(s)), s, ws, n)
             for s, ws, n in TEMPLATES if '_' not in s]


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------

def parse_wazn(term: str, vocalized: str | None = None) -> dict:
    """
    Parse the wazn of `term` itself. Never inherits from any cluster.

    Returns dict with: query_wazn, query_wazn_status, candidates,
    radicals_guess, normalized_form, note.
    """
    norm = normalize(term)
    matches = []
    for rx, skel, wazns, n_rad in _COMPILED:
        m = rx.match(norm)
        if m:
            radicals = list(m.groups())
            # reject if an extracted "radical" is a pure augment letter in a
            # position where the template already consumed augments — i.e.
            # allow weak letters as radicals (و ي ء are legal radicals).
            matches.append({'skeleton': skel, 'wazns': wazns,
                            'radicals': radicals})

    # Generic fallbacks (pure-C skeletons) only count when nothing
    # augment-marked matched: a template containing a literal augment
    # explains the string more specifically.
    specific = [m for m in matches if any(c != 'C' for c in m['skeleton'])]
    if specific:
        matches = specific

    if not matches:
        return {
            'query_wazn': None,
            'query_wazn_status': 'not_parsed',
            'candidates': [],
            'radicals_guess': None,
            'normalized_form': norm,
            'note': 'Surface form matches no template in inventory; '
                    'manual philological review required.',
        }

    # Prefer the template consuming the most structure (longest skeleton),
    # i.e. the most specific match.
    matches.sort(key=lambda m: len(m['skeleton']), reverse=True)
    best = matches[0]
    all_wazns = []
    for m in matches:
        for w in m['wazns']:
            if w not in all_wazns:
                all_wazns.append(w)

    if len(all_wazns) == 1:
        status = 'parsed'
        wazn = all_wazns[0]
    elif len(best['wazns']) > 1 and len(matches) == 1:
        status = 'ambiguous_vocalization'
        wazn = ' / '.join(best['wazns'])
    else:
        status = 'ambiguous_structure'
        wazn = ' / '.join(all_wazns[:4])

    # If caller supplied a vocalized form, try to disambiguate.
    if vocalized and status != 'parsed':
        narrowed = _narrow_by_vowels(vocalized, all_wazns)
        if len(narrowed) == 1:
            wazn, status = narrowed[0], 'parsed'
        elif narrowed:
            wazn, status = ' / '.join(narrowed), 'ambiguous_vocalization'

    return {
        'query_wazn': wazn,
        'query_wazn_status': status,
        'candidates': all_wazns,
        'radicals_guess': best['radicals'],
        'normalized_form': norm,
        'note': 'Parsed from the query term itself (miqyās positional '
                'substitution). Cluster wazn, if any, is reported '
                'separately and is NOT this field.',
    }


_VOWEL_MAP = {  # first two short vowels → pattern family discriminator
    ('\u064E', '\u0652'): ['faʿl'],            # fatḥa, sukūn
    ('\u0650', '\u0652'): ['fiʿl'],            # kasra, sukūn
    ('\u064F', '\u0652'): ['fuʿl'],            # ḍamma, sukūn
    ('\u064E', '\u064E'): ['faʿal'],
    ('\u064E', '\u0650'): ['faʿil'],
    ('\u064E', '\u064F'): ['faʿul', 'faʿūl'],
    ('\u064F', '\u064F'): ['fuʿūl', 'fuʿul'],
}


def _narrow_by_vowels(vocalized: str, candidates: list) -> list:
    marks = [c for c in vocalized if c in '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652']
    if len(marks) < 2:
        return candidates
    key = (marks[0], marks[1])
    fam = _VOWEL_MAP.get(key)
    if not fam:
        return candidates
    narrowed = [c for c in candidates if any(c.startswith(f.split('ʿ')[0][:2]) or c == f
                                             for f in fam)]
    return narrowed or candidates


# ----------------------------------------------------------------------
# Server integration shim
# ----------------------------------------------------------------------

def lookup_output_fields(term: str, cluster_wazn: str | None) -> dict:
    """Drop-in replacement for the old `dominant_wazn` assignment.

    OLD (bug):  record['dominant_wazn'] = attractor_cluster_majority_wazn
    NEW:        record.update(lookup_output_fields(term, cluster_majority))
    """
    p = parse_wazn(term)
    return {
        'query_wazn': p['query_wazn'],
        'query_wazn_status': p['query_wazn_status'],
        'cluster_wazn': cluster_wazn,   # kept, but explicitly labeled
    }


if __name__ == '__main__':
    # Regression suite: every live-confirmed bug instance from the
    # 2026-06-10 tool test, plus controls.
    cases = [
        ('مرآة',  'mifʿalah family'),     # bug: reported fāʿil
        ('نسيان', 'fiʿlān family'),       # bug: reported faʿl
        ('ذكر',   'CCC ambiguous'),       # bug: reported afʿal
        ('حرف',   'CCC ambiguous'),       # bug: reported fuʿūl (from Nūr cluster)
        ('حدس',   'CCC ambiguous'),       # bug: reported faʿīl
        ('عتبة',  'faʿlah family'),       # bug: reported faʿīl
        ('سكينة', 'faʿīlah'),             # coincidentally near-right before
        ('سكوت',  'faʿūl/fuʿūl'),
        ('كاتب',  'fāʿil'),
        ('مكتوب', 'mafʿūl'),
        ('استحضار', 'istifʿāl'),
    ]
    for t, expect in cases:
        r = parse_wazn(t)
        print(f"{t:>10}  →  {r['query_wazn']}  [{r['query_wazn_status']}]"
              f"   (expect: {expect})")
