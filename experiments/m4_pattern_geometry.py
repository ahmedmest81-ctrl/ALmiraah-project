"""
AL-MIR'ĀH Research Framework
Measurement 4: Pattern-Geometry — The Wazn as Geometric Operator
=================================================================

Ibn Arabi's insight formalised:
  The Arabic morphological pattern (وزن, wazn) is not a suffix or an affix —
  it is an abstract transformation applied uniformly across the root space.

  كَتَبَ  →  كَاتِب   →  مَكْتُوب
  ظَلَمَ  →  ظَالِم   →  مَظْلُوم
  فَتَحَ  →  فَاتِح   →  مَفْتُوح

  If the wazn is geometrically real, then:
    vec(كاتب) − vec(كتب)  ≈  vec(ظالم) − vec(ظلم)  ≈  vec(فاتح) − vec(فتح)

  The offset vector (V1 → AP) should be CONSISTENT across all 15 roots.
  The same holds for V1 → PP, V1 → VN, etc.

Manual annotation:
  Every word in the 150-word Q1 lexicon is tagged with its pattern code
  (wazn). These tags are the "anchors" — fixed coordinates in a
  (root × pattern) grid that is defined by morphological analysis,
  not by distributional statistics.

  Pattern codes:
    V1    = Form I base verb           (كَتَبَ, ظَلَمَ)
    AP    = Active participle fā'il    (كَاتِب, ظَالِم)  ← the agent
    PP    = Passive participle maf'ūl  (مَكْتُوب, مَظْلُوم)  ← the patient
    VN    = Verbal noun / maṣdar       (كِتَاب, ظُلْم)
    PN    = Place / instrument noun    (مَكْتَب, مَظْلَمَة)
    PL    = Broken plural              (كُتُب, مَظَالِم)
    V2    = Form II  fa''ala           (كَتَّبَ, عَلَّمَ)
    V2AP  = Form II active participle  (مُعَلِّم, مُدَرِّس)
    V2PP  = Form II passive participle (مُحَمَّل)
    V4    = Form IV  af'ala            (أَخْرَجَ, أَرْجَعَ)
    V4AP  = Form IV active participle  (مُخْرِج, مُصْلِح)
    V5    = Form V   tafa''ala         (تَعَلَّمَ, تَخَرَّجَ)
    V5AP  = Form V active participle   (مُتَعَلِّم)
    V6    = Form VI  tafā'ala          (تَرَاجَعَ, تَنَاظَرَ)
    V8    = Form VIII  ifta'ala        (اكْتَتَبَ, احْتَمَلَ)
    V10   = Form X   istaf'ala         (اسْتَخْرَجَ, اسْتَمَعَ)
    ADJ   = Intensive adjective / nisba (نَظَرِي, سَمِيع)
    OTHER = Other derivations

Tests:
  1. ANALOGY CONSISTENCY — for each pattern p, how consistent is the
     offset vector (vec_p − vec_V1) across all roots?
     Measured as: mean pairwise cosine similarity of offset vectors.
     If consistency ≈ 1.0 → pattern is a single geometric operator.
     If consistency ≈ 0.0 → pattern offsets are random per root.

  2. ANALOGY COMPLETION — leave-one-out prediction:
     predict vec(root_p) = vec(root_V1) + mean_offset(p, other roots)
     Measure: cosine similarity between predicted and actual vector.
     Also report: is actual word the nearest neighbour of the prediction?

  3. PATTERN CLUSTERING — do all AP words cluster together regardless
     of which root they come from?
     within-pattern similarity vs cross-pattern similarity.

  4. PCA VISUALISATION — plot all 150 words coloured by (a) root and
     (b) pattern. If the pattern is a geometric operator, pattern
     coloring should show clear structure independent of root coloring.

Usage:
  python q1_experiment/m4_pattern_geometry.py [--output m4_results]
  python q1_experiment/m4_pattern_geometry.py --model aravec \\
      --arabic "C:/path/to/full_grams_cbow_300_twitter.mdl"
"""

import argparse
import json
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PROJECT_ROOT)
from q1_root_cluster_density import strip_diacritics

ARABIC_CLASSICAL_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-ca"

ARABIC_CARRIERS = [
    "هذه الكلمة هي {} .",
    "المعنى الأساسي لكلمة {} واضح في اللغة العربية .",
    "تُعدّ {} من الكلمات الشائعة في النصوص العربية الكلاسيكية .",
]

# ── Manual annotation: (root, word_with_diacritics, pattern_code, description) ──
# These are the ANCHORS — positions in the (root × wazn) grid defined by
# morphological analysis, not by distributional statistics.

WORD_ANNOTATIONS = [
    # ── ك-ت-ب  (writing) ──────────────────────────────────────────────────────
    ("ك-ت-ب", "كَتَبَ",       "V1",   "Form I verb — he wrote"),
    ("ك-ت-ب", "كِتَاب",      "VN",   "Verbal noun fi'āl — book/writing"),
    ("ك-ت-ب", "كَاتِب",      "AP",   "Active participle fā'il — writer"),
    ("ك-ت-ب", "مَكْتُوب",    "PP",   "Passive participle maf'ūl — written/letter"),
    ("ك-ت-ب", "كِتَابَات",   "PL",   "Broken plural — writings"),
    ("ك-ت-ب", "مَكْتَب",     "PN",   "Place noun maf'al — office/desk"),
    ("ك-ت-ب", "كُتُب",       "PL",   "Broken plural fu'ul — books"),
    ("ك-ت-ب", "كَتَّبَ",     "V2",   "Form II fa''ala — made to write"),
    ("ك-ت-ب", "اكْتَتَبَ",   "V8",   "Form VIII ifta'ala — subscribed"),
    ("ك-ت-ب", "كِتَابَه",    "VN",   "Verbal noun fi'āla — his writing"),
    # ── ع-ل-م  (knowledge) ────────────────────────────────────────────────────
    ("ع-ل-م", "عَلِمَ",      "V1",   "Form I verb — he knew"),
    ("ع-ل-م", "عِلْم",       "VN",   "Verbal noun fi'l — knowledge"),
    ("ع-ل-م", "عَالِم",      "AP",   "Active participle fā'il — scholar"),
    ("ع-ل-م", "مَعْلُوم",    "PP",   "Passive participle maf'ūl — known"),
    ("ع-ل-م", "تَعَلَّمَ",   "V5",   "Form V tafa''ala — he learned"),
    ("ع-ل-م", "عَلَّمَ",     "V2",   "Form II fa''ala — he taught"),
    ("ع-ل-م", "مُعَلِّم",    "V2AP", "Form II active participle — teacher"),
    ("ع-ل-م", "مُتَعَلِّم",  "V5AP", "Form V active participle — learner"),
    ("ع-ل-م", "عُلُوم",      "PL",   "Broken plural — sciences"),
    ("ع-ل-م", "مَعْلُومَات", "PL",   "Plural of maf'ūl — information"),
    # ── د-ر-س  (study) ────────────────────────────────────────────────────────
    ("د-ر-س", "دَرَسَ",      "V1",   "Form I verb — he studied"),
    ("د-ر-س", "دِرَاسَه",    "VN",   "Verbal noun fi'āla — studying"),
    ("د-ر-س", "دَارِس",      "AP",   "Active participle fā'il — student"),
    ("د-ر-س", "مَدْرَسَه",   "PN",   "Place noun maf'ala — school"),
    ("د-ر-س", "دَرَّسَ",     "V2",   "Form II — he taught"),
    ("د-ر-س", "مُدَرِّس",    "V2AP", "Form II active participle — teacher"),
    ("د-ر-س", "دُرُوس",      "PL",   "Broken plural — lessons"),
    ("د-ر-س", "مَدْرُوس",    "PP",   "Passive participle maf'ūl — studied"),
    ("د-ر-س", "دِرَاسِي",    "ADJ",  "Nisba adjective — academic"),
    ("د-ر-س", "دِرَاسَة",    "VN",   "Verbal noun fi'āla — study"),
    # ── ق-ر-أ  (reading) ──────────────────────────────────────────────────────
    ("ق-ر-أ", "قَرَا",       "V1",   "Form I verb — he read"),
    ("ق-ر-أ", "قِرَاءَه",    "VN",   "Verbal noun fi'āla — reading"),
    ("ق-ر-أ", "قَارِئ",      "AP",   "Active participle fā'il — reader"),
    ("ق-ر-أ", "مَقْرُوء",    "PP",   "Passive participle maf'ūl — that which is read"),
    ("ق-ر-أ", "قِرَاءَات",   "PL",   "Plural — readings"),
    ("ق-ر-أ", "قُرَّاء",     "PL",   "Broken plural of fā'il — readers"),
    ("ق-ر-أ", "قَارِئِين",   "PL",   "Regular plural — readers"),
    ("ق-ر-أ", "قِرَائِه",    "VN",   "Verbal noun possessive — his reading"),
    ("ق-ر-أ", "مَقْرُوءَة",  "PP",   "Passive participle feminine"),
    ("ق-ر-أ", "قِرَاءَات",   "PL",   "Plural — readings (variant)"),
    # ── خ-ر-ج  (exit) ─────────────────────────────────────────────────────────
    ("خ-ر-ج", "خَرَجَ",      "V1",   "Form I verb — he went out"),
    ("خ-ر-ج", "خُرُوج",      "VN",   "Verbal noun fu'ūl — going out"),
    ("خ-ر-ج", "خَارِج",      "AP",   "Active participle fā'il — outside/exiting"),
    ("خ-ر-ج", "مَخْرَج",     "PN",   "Place noun maf'al — exit"),
    ("خ-ر-ج", "أَخْرَجَ",    "V4",   "Form IV af'ala — he took out"),
    ("خ-ر-ج", "تَخَرَّجَ",   "V5",   "Form V — he graduated"),
    ("خ-ر-ج", "اسْتَخْرَجَ", "V10",  "Form X — he extracted"),
    ("خ-ر-ج", "خِرِّيج",     "OTHER","Graduate (fi''īl intensive pattern)"),
    ("خ-ر-ج", "خَرَّجَ",     "V2",   "Form II — he graduated/expelled"),
    ("خ-ر-ج", "مُخْرِج",     "V4AP", "Form IV active participle — director"),
    # ── ر-ج-ع  (return) ───────────────────────────────────────────────────────
    ("ر-ج-ع", "رَجَعَ",      "V1",   "Form I verb — he returned"),
    ("ر-ج-ع", "رُجُوع",      "VN",   "Verbal noun fu'ūl — return"),
    ("ر-ج-ع", "رَاجِع",      "AP",   "Active participle fā'il — returning/reviewing"),
    ("ر-ج-ع", "مَرْجِع",     "PN",   "Place/source noun maf'il — reference"),
    ("ر-ج-ع", "أَرْجَعَ",    "V4",   "Form IV — he returned sth"),
    ("ر-ج-ع", "تَرَاجَعَ",   "V6",   "Form VI tafā'ala — he receded"),
    ("ر-ج-ع", "اسْتَرْجَعَ", "V10",  "Form X — he retrieved"),
    ("ر-ج-ع", "رَجْعَة",     "VN",   "Verbal noun fa'la — a return"),
    ("ر-ج-ع", "مَرْجِعِي",   "ADJ",  "Nisba adjective — referential"),
    ("ر-ج-ع", "رَاجَعَ",     "OTHER","Form III fā'ala — he reviewed"),
    # ── ح-م-ل  (carry) ────────────────────────────────────────────────────────
    ("ح-م-ل", "حَمَلَ",      "V1",   "Form I verb — he carried"),
    ("ح-م-ل", "حَمْل",       "VN",   "Verbal noun fa'l — carrying / pregnancy"),
    ("ح-م-ل", "حَامِل",      "AP",   "Active participle fā'il — carrier / pregnant"),
    ("ح-م-ل", "مَحْمُول",    "PP",   "Passive participle maf'ūl — carried / mobile"),
    ("ح-م-ل", "أَحْمَلَ",    "V4",   "Form IV — he made carry"),
    ("ح-م-ل", "تَحَمَّلَ",   "V5",   "Form V — he bore / endured"),
    ("ح-م-ل", "احْتَمَلَ",   "V8",   "Form VIII — he tolerated / was probable"),
    ("ح-م-ل", "حِمْل",       "VN",   "Verbal noun fi'l — load / burden"),
    ("ح-م-ل", "حَمَّالَة",   "OTHER","Strap / carrier — fa''āla pattern"),
    ("ح-م-ل", "مُحَمَّل",    "V2PP", "Form II passive participle — laden"),
    # ── ف-ت-ح  (open) ─────────────────────────────────────────────────────────
    ("ف-ت-ح", "فَتَحَ",      "V1",   "Form I verb — he opened"),
    ("ف-ت-ح", "فَتْح",       "VN",   "Verbal noun fa'l — opening / conquest"),
    ("ف-ت-ح", "فَاتِح",      "AP",   "Active participle fā'il — opener / conqueror"),
    ("ف-ت-ح", "مَفْتُوح",    "PP",   "Passive participle maf'ūl — open"),
    ("ف-ت-ح", "فَتَّحَ",     "V2",   "Form II — he opened wide"),
    ("ف-ت-ح", "انْفَتَحَ",   "OTHER","Form VII infa'ala — it opened"),
    ("ف-ت-ح", "افْتَتَحَ",   "V8",   "Form VIII — he inaugurated"),
    ("ف-ت-ح", "فِتَاح",      "VN",   "Verbal noun fi'āl — opening"),
    ("ف-ت-ح", "مَفْتَاح",    "PN",   "Instrument noun mif'āl — key"),
    ("ف-ت-ح", "فَتَّاح",     "ADJ",  "Intensive agent fa''āl — the Opener (divine name)"),
    # ── ص-ل-ح  (righteousness / reform) ──────────────────────────────────────
    ("ص-ل-ح", "صَلَحَ",      "V1",   "Form I verb — he was good / suitable"),
    ("ص-ل-ح", "صَلَاح",      "VN",   "Verbal noun fa'āl — righteousness"),
    ("ص-ل-ح", "صَالِح",      "AP",   "Active participle fā'il — righteous / good"),
    ("ص-ل-ح", "مَصْلَحَه",   "PN",   "Place noun maf'ala — welfare / interest"),
    ("ص-ل-ح", "اصْلَحَ",     "V4",   "Form IV af'ala — he reformed"),
    ("ص-ل-ح", "اسْتَصْلَحَ", "V10",  "Form X — he sought reform"),
    ("ص-ل-ح", "اصْلَاح",     "VN",   "Form IV verbal noun if'āl — reform"),
    ("ص-ل-ح", "مُصْلِح",     "V4AP", "Form IV active participle — reformer"),
    ("ص-ل-ح", "صَلُوح",      "ADJ",  "Intensive fa'ūl — very righteous"),
    ("ص-ل-ح", "تَصَالَحَ",   "V6",   "Form VI tafā'ala — they reconciled"),
    # ── ق-و-ل  (speech / saying) ──────────────────────────────────────────────
    ("ق-و-ل", "قَالَ",       "V1",   "Form I verb — he said"),
    ("ق-و-ل", "قَوْل",       "VN",   "Verbal noun fa'l — saying / statement"),
    ("ق-و-ل", "قَائِل",      "AP",   "Active participle fā'il — one who says"),
    ("ق-و-ل", "مَقُول",      "PP",   "Passive participle maf'ūl — said / spoken"),
    ("ق-و-ل", "أَقْوَال",    "PL",   "Broken plural af'āl — sayings"),
    ("ق-و-ل", "قَوَّلَ",     "V2",   "Form II — he put words in mouth"),
    ("ق-و-ل", "تَقَوَّلَ",   "V5",   "Form V — he fabricated / attributed falsely"),
    ("ق-و-ل", "مَقَال",      "VN",   "maf'āl — article / essay"),
    ("ق-و-ل", "قِيل",        "OTHER","Passive of V1 — 'it was said'"),
    ("ق-و-ل", "مَقُولَة",    "PP",   "Passive participle feminine — maxim"),
    # ── ع-م-ل  (work / action) ────────────────────────────────────────────────
    ("ع-م-ل", "عَمِلَ",      "V1",   "Form I verb — he worked"),
    ("ع-م-ل", "عَمَل",       "VN",   "Verbal noun fa'al — work / deed"),
    ("ع-م-ل", "عَامِل",      "AP",   "Active participle fā'il — worker / agent"),
    ("ع-م-ل", "مَعْمُول",    "PP",   "Passive participle maf'ūl — made / object"),
    ("ع-م-ل", "أَعْمَال",    "PL",   "Broken plural af'āl — works / deeds"),
    ("ع-م-ل", "عَمَّلَ",     "V2",   "Form II — he employed"),
    ("ع-م-ل", "اعْتَمَلَ",   "V8",   "Form VIII — he worked hard"),
    ("ع-م-ل", "عَمَلِي",     "ADJ",  "Nisba adjective — practical / applied"),
    ("ع-م-ل", "مَعْمَل",     "PN",   "Place noun maf'al — laboratory / factory"),
    ("ع-م-ل", "عُمَّال",     "PL",   "Broken plural — workers"),
    # ── ن-ظ-ر  (sight / theory) ───────────────────────────────────────────────
    ("ن-ظ-ر", "نَظَرَ",      "V1",   "Form I verb — he looked"),
    ("ن-ظ-ر", "نَظَر",       "VN",   "Verbal noun fa'al — sight / view"),
    ("ن-ظ-ر", "نَاظِر",      "AP",   "Active participle fā'il — gazer / overseer"),
    ("ن-ظ-ر", "مَنْظُور",    "PP",   "Passive participle maf'ūl — viewed / perspective"),
    ("ن-ظ-ر", "أَنْظَار",    "PL",   "Broken plural af'āl — gazes / views"),
    ("ن-ظ-ر", "نَظَّرَ",     "V2",   "Form II — he theorized"),
    ("ن-ظ-ر", "تَنَاظَرَ",   "V6",   "Form VI — they faced each other / debated"),
    ("ن-ظ-ر", "مِنْظَار",    "PN",   "Instrument noun mif'āl — telescope / prism"),
    ("ن-ظ-ر", "نَظَرِي",     "ADJ",  "Nisba adjective — theoretical"),
    ("ن-ظ-ر", "مَنْظَر",     "PN",   "Place noun maf'al — view / scene"),
    # ── ب-ح-ث  (research / inquiry) ───────────────────────────────────────────
    ("ب-ح-ث", "بَحَثَ",      "V1",   "Form I verb — he researched"),
    ("ب-ح-ث", "بَحْث",       "VN",   "Verbal noun fa'l — research"),
    ("ب-ح-ث", "بَاحِث",      "AP",   "Active participle fā'il — researcher"),
    ("ب-ح-ث", "مَبْحُوث",    "PP",   "Passive participle maf'ūl — researched"),
    ("ب-ح-ث", "أَبْحَاث",    "PL",   "Broken plural af'āl — researches"),
    ("ب-ح-ث", "بَحَّثَ",     "V2",   "Form II — he investigated thoroughly"),
    ("ب-ح-ث", "ابْتَحَثَ",   "V8",   "Form VIII — he researched"),
    ("ب-ح-ث", "بَحْثِي",     "ADJ",  "Nisba adjective — research / academic"),
    ("ب-ح-ث", "مَبْحَث",     "PN",   "Place noun maf'al — research topic"),
    ("ب-ح-ث", "بُحُوث",      "PL",   "Broken plural fu'ūl — researches"),
    # ── س-م-ع  (hearing) ──────────────────────────────────────────────────────
    ("س-م-ع", "سَمِعَ",      "V1",   "Form I verb — he heard"),
    ("س-م-ع", "سَمَاع",      "VN",   "Verbal noun fa'āl — hearing / audition"),
    ("س-م-ع", "سَامِع",      "AP",   "Active participle fā'il — hearer / listener"),
    ("س-م-ع", "مَسْمُوع",    "PP",   "Passive participle maf'ūl — heard"),
    ("س-م-ع", "اسْمَعَ",     "OTHER","Imperative of Form IV — let him hear"),
    ("س-م-ع", "اسْتَمَعَ",   "V10",  "Form X — he listened attentively"),
    ("س-م-ع", "سَمَاعَه",    "VN",   "Verbal noun possessive / earphone"),
    ("س-م-ع", "سَمِيع",      "ADJ",  "Intensive fa'īl — all-hearing (divine name)"),
    ("س-م-ع", "مِسْمَع",     "PN",   "Instrument noun mif'al — hearing range"),
    ("س-م-ع", "سَمَّاعَة",   "OTHER","Earphone / receiver — fa''āla pattern"),
    # ── ج-ل-س  (sitting / council) ────────────────────────────────────────────
    ("ج-ل-س", "جَلَسَ",      "V1",   "Form I verb — he sat"),
    ("ج-ل-س", "جُلُوس",      "VN",   "Verbal noun fu'ūl — sitting"),
    ("ج-ل-س", "جَالِس",      "AP",   "Active participle fā'il — sitting / seated"),
    ("ج-ل-س", "مَجْلِس",     "PN",   "Place noun maf'il — council / sitting place"),
    ("ج-ل-س", "اجْلَسَ",     "V4",   "Form IV — he seated someone"),
    ("ج-ل-س", "جَلَّسَ",     "V2",   "Form II — he made sit"),
    ("ج-ل-س", "اجْتَلَسَ",   "V8",   "Form VIII — they sat together"),
    ("ج-ل-س", "جِلْسَه",     "VN",   "Verbal noun fi'la — sitting session"),
    ("ج-ل-س", "مُجَالِس",    "OTHER","Form III active participle — sitting companion"),
    ("ج-ل-س", "جُلَسَاء",    "PL",   "Broken plural fu'alā' — sitting companions"),
]

# Primary triad — the three patterns present in all 15 roots.
# These are the cleanest test of Ibn Arabi's geometric operator hypothesis.
PRIMARY_TRIAD = ["V1", "AP", "PP"]

# English morphological parallel for comparison
ENGLISH_ANNOTATIONS = [
    # (family, word, pattern_code, description)
    ("write",  "write",       "V1",  "Base verb"),
    ("write",  "writing",     "VN",  "Gerund / verbal noun"),
    ("write",  "writer",      "AP",  "Agent noun -er"),
    ("write",  "written",     "PP",  "Past participle"),
    ("write",  "writings",    "PL",  "Plural"),
    ("write",  "rewrite",     "V_MOD","Modified verb"),
    ("write",  "overwrite",   "V_MOD","Modified verb"),
    ("know",   "know",        "V1",  "Base verb"),
    ("know",   "knowing",     "VN",  "Gerund"),
    ("know",   "knower",      "AP",  "Agent noun -er"),
    ("know",   "known",       "PP",  "Past participle"),
    ("know",   "knowledge",   "VN",  "Verbal noun (irregular)"),
    ("know",   "unknown",     "PP",  "Negated participle"),
    ("study",  "study",       "V1",  "Base verb"),
    ("study",  "studying",    "VN",  "Gerund"),
    ("study",  "student",     "AP",  "Agent noun (irregular)"),
    ("study",  "studied",     "PP",  "Past participle"),
    ("study",  "studies",     "PL",  "Plural / third person"),
    ("read",   "read",        "V1",  "Base verb"),
    ("read",   "reading",     "VN",  "Gerund"),
    ("read",   "reader",      "AP",  "Agent noun -er"),
    ("work",   "work",        "V1",  "Base verb"),
    ("work",   "working",     "VN",  "Gerund"),
    ("work",   "worker",      "AP",  "Agent noun -er"),
    ("work",   "worked",      "PP",  "Past participle"),
    ("speak",  "speak",       "V1",  "Base verb"),
    ("speak",  "speaking",    "VN",  "Gerund"),
    ("speak",  "speaker",     "AP",  "Agent noun -er"),
    ("speak",  "spoken",      "PP",  "Past participle"),
    ("teach",  "teach",       "V1",  "Base verb"),
    ("teach",  "teaching",    "VN",  "Gerund"),
    ("teach",  "teacher",     "AP",  "Agent noun -er"),
    ("teach",  "taught",      "PP",  "Past participle (irregular)"),
    ("open",   "open",        "V1",  "Base verb"),
    ("open",   "opening",     "VN",  "Gerund"),
    ("open",   "opener",      "AP",  "Agent noun -er"),
    ("open",   "opened",      "PP",  "Past participle"),
    ("see",    "see",         "V1",  "Base verb"),
    ("see",    "seeing",      "VN",  "Gerund"),
    ("see",    "seer",        "AP",  "Agent noun -er"),
    ("see",    "seen",        "PP",  "Past participle"),
]

ENGLISH_CARRIERS = [
    "This word is {} .",
    "The meaning of {} is well known in the English language .",
    "{} is a common word used in everyday English .",
]


# ── Embedding helpers ─────────────────────────────────────────────────────────

def load_bert(model_name):
    from transformers import AutoTokenizer, AutoModel
    print(f"  Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    return tok, mdl


def get_contextual_vector(tokenizer, model, word, carrier):
    import torch
    inputs = tokenizer(carrier, return_tensors="pt", truncation=True, max_length=64)
    word_tokens = tokenizer.tokenize(word)
    all_tokens  = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    for i in range(len(all_tokens) - len(word_tokens) + 1):
        if all_tokens[i:i + len(word_tokens)] == word_tokens:
            idx = list(range(i, i + len(word_tokens)))
            with __import__('torch').no_grad():
                out = model(**inputs)
            return out.last_hidden_state[0][idx].mean(dim=0).numpy()
    return None


def get_averaged_vector(tokenizer, model, word, carriers):
    vecs = [v for tpl in carriers
            if (v := get_contextual_vector(tokenizer, model, word,
                                           tpl.format(word))) is not None]
    return np.mean(vecs, axis=0) if vecs else None


def load_aravec(path):
    from gensim.models import Word2Vec
    print(f"  Loading AraVec from {path} ...")
    wv = Word2Vec.load(path).wv
    print(f"  Vocab: {len(wv.key_to_index):,}")
    return wv


def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0


# ── Build annotated vector table ──────────────────────────────────────────────

def embed_annotated_lexicon(annotations, tokenizer, model, carriers,
                             preprocess=None):
    """
    Embed every annotated word. Returns list of dicts:
      {family, word, pattern, description, vector}
    """
    table = []
    missing = []
    seen = set()   # skip exact duplicates (e.g. duplicate قِرَاءَات entry)
    for family, raw_word, pattern, desc in annotations:
        word = preprocess(raw_word) if preprocess else raw_word
        key  = (family, word, pattern)
        if key in seen:
            continue
        seen.add(key)
        vec = get_averaged_vector(tokenizer, model, word, carriers)
        if vec is not None:
            table.append({'family': family, 'word': word, 'raw': raw_word,
                          'pattern': pattern, 'description': desc,
                          'vector': vec})
        else:
            missing.append(word)
    print(f"  Embedded {len(table)} words | missing: {len(missing)}"
          + (f" {missing[:5]}" if missing else ""))
    return table


def embed_aravec_lexicon(annotations, wv, preprocess=None):
    """Same structure but using static AraVec vectors."""
    table = []
    missing = []
    seen = set()
    for family, raw_word, pattern, desc in annotations:
        word = preprocess(raw_word) if preprocess else raw_word
        key  = (family, word, pattern)
        if key in seen:
            continue
        seen.add(key)
        try:
            vec = wv[word]
            table.append({'family': family, 'word': word, 'raw': raw_word,
                          'pattern': pattern, 'description': desc,
                          'vector': vec})
        except KeyError:
            missing.append(word)
    print(f"  Embedded {len(table)} words | missing: {len(missing)}"
          + (f" {missing[:5]}" if missing else ""))
    return table


# ── Core geometric analysis ───────────────────────────────────────────────────

def get_family_pattern_map(table):
    """
    Returns {family: {pattern: list_of_vectors}}.
    """
    from collections import defaultdict
    fpm = defaultdict(lambda: defaultdict(list))
    for row in table:
        fpm[row['family']][row['pattern']].append(row['vector'])
    # Collapse lists to single vectors (mean if multiple)
    return {f: {p: np.mean(vecs, axis=0) for p, vecs in pats.items()}
            for f, pats in fpm.items()}


def compute_pattern_offsets(fpm, anchor_pattern="V1"):
    """
    For each family that has both anchor_pattern and at least one other pattern,
    compute offset = vec(other_pattern) - vec(anchor_pattern).
    Returns {pattern: list_of_offset_vectors_across_families}.
    """
    from collections import defaultdict
    offsets = defaultdict(list)
    for family, pats in fpm.items():
        if anchor_pattern not in pats:
            continue
        anchor_vec = pats[anchor_pattern]
        for pattern, vec in pats.items():
            if pattern == anchor_pattern:
                continue
            offsets[pattern].append(vec - anchor_vec)
    return dict(offsets)


def analogy_consistency(offsets):
    """
    For each pattern, compute mean pairwise cosine similarity of its offset
    vectors across families. High consistency → pattern is a stable operator.
    Returns {pattern: {'n': int, 'mean_consistency': float, 'std': float}}
    """
    results = {}
    for pattern, vecs in offsets.items():
        if len(vecs) < 2:
            continue
        sims = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                sims.append(cosine_sim(vecs[i], vecs[j]))
        results[pattern] = {
            'n_families': len(vecs),
            'mean_consistency': float(np.mean(sims)),
            'std_consistency':  float(np.std(sims)),
            'min': float(np.min(sims)),
            'max': float(np.max(sims)),
        }
    return results


def analogy_completion(fpm, offsets, anchor_pattern="V1"):
    """
    Leave-one-out analogy completion:
      For root r and pattern p, predict:
        pred = vec(r, V1) + mean_offset(p, excluding r)
      Measure cosine similarity between pred and actual vec(r, p).

    Returns list of {'family', 'pattern', 'predicted_sim', 'rank_in_all_words'}.
    """
    # Flat table for rank calculation
    all_vecs   = []
    all_labels = []
    for family, pats in fpm.items():
        for pat, vec in pats.items():
            all_vecs.append(vec)
            all_labels.append(f"{family}:{pat}")
    all_vecs = np.array(all_vecs)

    results = []
    for family, pats in fpm.items():
        if anchor_pattern not in pats:
            continue
        anchor_vec = pats[anchor_pattern]
        for pattern, target_vec in pats.items():
            if pattern == anchor_pattern:
                continue
            if pattern not in offsets or len(offsets[pattern]) < 2:
                continue
            # Leave-one-out: exclude this family's offset
            this_offset  = target_vec - anchor_vec
            other_offsets = [v for i, (f, pats2) in enumerate(fpm.items())
                             if f != family and pattern in pats2
                             for v in [pats2[pattern] - pats2[anchor_pattern]
                                       if anchor_pattern in pats2 else None]
                             if v is not None]
            if not other_offsets:
                continue
            mean_offset = np.mean(other_offsets, axis=0)
            predicted   = anchor_vec + mean_offset

            # Cosine similarity between prediction and actual
            pred_sim = cosine_sim(predicted, target_vec)

            # Rank of actual vector among all vectors by similarity to prediction
            all_sims = [cosine_sim(predicted, v) for v in all_vecs]
            rank = 1 + sum(1 for s in all_sims if s > pred_sim)

            results.append({
                'family': family,
                'pattern': pattern,
                'predicted_sim': float(pred_sim),
                'rank_of_target': rank,
                'total_words': len(all_vecs),
                'target_word': next((r['word'] for r in []  # placeholder
                                     ), ''),
            })
    return results


def pattern_clustering(table, patterns_to_test=None):
    """
    For each pattern: compare within-pattern cosine similarity to
    cross-pattern cosine similarity.
    """
    from scipy import stats as scipy_stats

    if patterns_to_test is None:
        patterns_to_test = list(set(r['pattern'] for r in table))

    results = {}
    all_data = {r['pattern']: r['vector'] for r in table}  # last seen per pattern

    # Group by pattern
    by_pattern = {}
    for row in table:
        by_pattern.setdefault(row['pattern'], []).append(row['vector'])

    for pattern in patterns_to_test:
        if pattern not in by_pattern or len(by_pattern[pattern]) < 2:
            continue
        vecs = by_pattern[pattern]

        # Within-pattern pairs
        within = [cosine_sim(vecs[i], vecs[j])
                  for i in range(len(vecs)) for j in range(i+1, len(vecs))]

        # Cross-pattern pairs (sample)
        other_vecs = [v for p, pvecs in by_pattern.items()
                      if p != pattern for v in pvecs]
        if not other_vecs:
            continue
        rng = np.random.default_rng(42)
        n_cross = min(500, len(within) * 5)
        cross = [cosine_sim(vecs[rng.integers(len(vecs))],
                             other_vecs[rng.integers(len(other_vecs))])
                 for _ in range(n_cross)]

        gap = float(np.mean(within) - np.mean(cross))
        if len(within) >= 3 and len(cross) >= 3:
            _, p_val = scipy_stats.mannwhitneyu(within, cross, alternative='greater')
        else:
            p_val = 1.0

        results[pattern] = {
            'n_words': len(vecs),
            'within_mean': float(np.mean(within)),
            'within_std':  float(np.std(within)),
            'cross_mean':  float(np.mean(cross)),
            'cross_std':   float(np.std(cross)),
            'gap':         gap,
            'p_value':     float(p_val),
            'significant': bool(p_val < 0.05),
        }
    return results


# ── Visualizations ────────────────────────────────────────────────────────────

PATTERN_COLORS = {
    "V1":    "#e8a030",   # orange — the base verb
    "AP":    "#c8303a",   # red — the agent (فاعل)
    "PP":    "#3070c8",   # blue — the patient (مفعول)
    "VN":    "#7ab07a",   # green — verbal noun
    "PN":    "#a050c0",   # purple — place noun
    "PL":    "#888888",   # grey — plurals
    "V2":    "#e870a0",   # pink — Form II
    "V2AP":  "#c06080",   # dark pink
    "V2PP":  "#d09098",
    "V4":    "#50c8a0",
    "V4AP":  "#30a080",
    "V5":    "#b0a030",
    "V5AP":  "#908020",
    "V6":    "#606060",
    "V8":    "#d04020",
    "V10":   "#208060",
    "ADJ":   "#4090d0",
    "OTHER": "#cccccc",
}

FAMILY_MARKERS = ['o','s','^','v','D','P','*','X','h','<','>','p','H','8','d']


def generate_visualizations(table_ar, consistency_ar, completion_ar,
                             clustering_ar, table_en, consistency_en,
                             clustering_en, out_dir):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    os.makedirs(out_dir, exist_ok=True)

    # ── Plot 1: PCA of Arabic words, coloured by PATTERN ──────────────
    if table_ar:
        vecs = np.array([r['vector'] for r in table_ar])
        pca  = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vecs)
        patterns = [r['pattern'] for r in table_ar]
        families = [r['family'] for r in table_ar]
        all_families = sorted(set(families))

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            "PCA of Arabic Root-Pattern Lexicon — CAMeLBERT-ca\n"
            "AL-MIR'ĀH Measurement 4: Pattern Geometry",
            fontsize=13, fontweight='bold'
        )

        # Left: coloured by pattern
        ax = axes[0]
        seen_patterns = set()
        for i, (x, y) in enumerate(coords):
            pat = patterns[i]
            col = PATTERN_COLORS.get(pat, '#999')
            label = pat if pat not in seen_patterns else None
            ax.scatter(x, y, c=col, s=60, alpha=0.85, label=label,
                       edgecolors='white', linewidths=0.4)
            seen_patterns.add(pat)
        ax.legend(fontsize=7, ncol=2, loc='best',
                  title='Pattern (wazn)', title_fontsize=8)
        ax.set_title("Coloured by morphological pattern\n"
                     "Clusters here = pattern is a geometric operator",
                     fontsize=10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Right: coloured by root family
        ax = axes[1]
        family_colors = plt.cm.tab20(np.linspace(0, 1, len(all_families)))
        seen_fam = set()
        for i, (x, y) in enumerate(coords):
            fam = families[i]
            fi  = all_families.index(fam)
            col = family_colors[fi]
            label = fam if fam not in seen_fam else None
            ax.scatter(x, y, c=[col], s=60, alpha=0.8, label=label,
                       marker=FAMILY_MARKERS[fi % len(FAMILY_MARKERS)],
                       edgecolors='white', linewidths=0.4)
            seen_fam.add(fam)
        ax.legend(fontsize=6, ncol=2, loc='best',
                  title='Root', title_fontsize=8)
        ax.set_title("Coloured by root family\n"
                     "Clusters here = root is the primary geometry",
                     fontsize=10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        p1 = os.path.join(out_dir, "m4_pca_arabic.png")
        plt.savefig(p1, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {p1}")

    # ── Plot 2: Analogy consistency — primary triad ────────────────────
    if consistency_ar:
        triad = {p: v for p, v in consistency_ar.items()
                 if p in ['AP', 'PP', 'VN', 'PN', 'V2', 'V2AP', 'V4', 'V5', 'V8', 'V10']}
        if triad:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.suptitle(
                "Analogy Consistency: Mean Pairwise Cosine Sim of Offset Vectors\n"
                "offset = vec(pattern) − vec(V1)   |   Higher = more consistent geometric operator",
                fontsize=12, fontweight='bold'
            )
            labels = list(triad.keys())
            means  = [triad[p]['mean_consistency'] for p in labels]
            stds   = [triad[p]['std_consistency']  for p in labels]
            ns     = [triad[p]['n_families']        for p in labels]
            colors = [PATTERN_COLORS.get(p, '#999') for p in labels]

            bars = ax.bar(range(len(labels)), means, color=colors, alpha=0.85,
                          yerr=stds, capsize=5, error_kw={'linewidth':1.5})
            for i, (m, n) in enumerate(zip(means, ns)):
                ax.text(i, m + max(stds)*0.05 + 0.01,
                        f'{m:.3f}\n(n={n})', ha='center', va='bottom', fontsize=8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylabel("Mean cosine similarity between offset vectors", fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.axhline(0.5, color='#aaa', linestyle=':', linewidth=1,
                       label='chance level')
            ax.axhline(1.0, color='#555', linestyle='--', linewidth=1,
                       label='perfect consistency')
            ax.legend(fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            p2 = os.path.join(out_dir, "m4_analogy_consistency.png")
            plt.savefig(p2, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {p2}")

    # ── Plot 3: Analogy completion accuracy ────────────────────────────
    if completion_ar:
        by_pattern = {}
        for r in completion_ar:
            by_pattern.setdefault(r['pattern'], []).append(r['predicted_sim'])
        patterns_sorted = sorted(by_pattern.keys(),
                                 key=lambda p: np.mean(by_pattern[p]), reverse=True)
        if patterns_sorted:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.suptitle(
                "Analogy Completion — Leave-One-Out Prediction Accuracy\n"
                "cos(pred, actual)  |  pred = vec(V1) + mean_offset(pattern, other roots)",
                fontsize=12, fontweight='bold'
            )
            means  = [np.mean(by_pattern[p]) for p in patterns_sorted]
            stds   = [np.std(by_pattern[p])  for p in patterns_sorted]
            colors = [PATTERN_COLORS.get(p, '#999') for p in patterns_sorted]
            ax.bar(range(len(patterns_sorted)), means, color=colors, alpha=0.85,
                   yerr=stds, capsize=4, error_kw={'linewidth':1.2})
            for i, m in enumerate(means):
                ax.text(i, m + max(stds)*0.05 + 0.005,
                        f'{m:.3f}', ha='center', va='bottom', fontsize=8)
            ax.set_xticks(range(len(patterns_sorted)))
            ax.set_xticklabels(patterns_sorted, fontsize=9)
            ax.set_ylabel("Cosine similarity: predicted vs actual", fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            p3 = os.path.join(out_dir, "m4_analogy_completion.png")
            plt.savefig(p3, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {p3}")

    # ── Plot 4: Pattern clustering gap — Arabic vs English ─────────────
    if clustering_ar and clustering_en:
        shared = [p for p in ['V1', 'AP', 'PP', 'VN', 'PL']
                  if p in clustering_ar and p in clustering_en]
        if shared:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                "Pattern Clustering Gap (within − cross pattern similarity)\n"
                "Arabic (CAMeLBERT-ca) vs English (BERT-base)",
                fontsize=12, fontweight='bold'
            )
            for ax, clust, title, color in [
                (axes[0], clustering_ar, "Arabic (CAMeLBERT-ca)", "#e8a030"),
                (axes[1], clustering_en, "English (BERT-base)",   "#4a6e8a"),
            ]:
                pats = [p for p in shared if p in clust]
                gaps = [clust[p]['gap'] for p in pats]
                cols = [PATTERN_COLORS.get(p, color) for p in pats]
                ax.bar(range(len(pats)), gaps, color=cols, alpha=0.85)
                for i, g in enumerate(gaps):
                    ax.text(i, g + 0.001, f'{g:.3f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
                ax.set_xticks(range(len(pats)))
                ax.set_xticklabels(pats, fontsize=10)
                ax.set_ylabel("Gap (within-pattern − cross-pattern sim)", fontsize=9)
                ax.set_title(title, fontsize=11)
                ax.axhline(0, color='black', linewidth=0.8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            plt.tight_layout()
            p4 = os.path.join(out_dir, "m4_pattern_clustering_gap.png")
            plt.savefig(p4, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {p4}")

    # ── Plot 5: Offset vectors for V1→AP across all 15 roots ──────────
    if table_ar:
        fpm = get_family_pattern_map(table_ar)
        all_fams = sorted(fpm.keys())
        triads_available = [(f, fpm[f]) for f in all_fams
                            if all(p in fpm[f] for p in PRIMARY_TRIAD)]
        if len(triads_available) >= 3:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                "Offset Vectors Visualised via PCA: V1 → AP and V1 → PP\n"
                "Each arrow = one root's transformation in embedding space.\n"
                "Parallel arrows = consistent geometric operator.",
                fontsize=11, fontweight='bold'
            )
            all_offset_vecs = []
            for fam, pats in triads_available:
                for p in PRIMARY_TRIAD:
                    all_offset_vecs.append(pats[p])
            pca2 = PCA(n_components=2, random_state=42)
            pca2.fit(np.array(all_offset_vecs))

            for ax, target_pat, title in [
                (axes[0], "AP", "V1 → AP  (verb → active agent فاعل)"),
                (axes[1], "PP", "V1 → PP  (verb → passive patient مفعول)"),
            ]:
                fc = plt.cm.tab20(np.linspace(0, 1, len(triads_available)))
                for fi, (fam, pats) in enumerate(triads_available):
                    if "V1" not in pats or target_pat not in pats:
                        continue
                    v1_2d  = pca2.transform([pats["V1"]])[0]
                    tgt_2d = pca2.transform([pats[target_pat]])[0]
                    ax.annotate("", xy=tgt_2d, xytext=v1_2d,
                                arrowprops=dict(arrowstyle="->",
                                                color=fc[fi], lw=1.5))
                    ax.text(*v1_2d, fam, fontsize=6, color=fc[fi], alpha=0.8)
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            plt.tight_layout()
            p5 = os.path.join(out_dir, "m4_offset_arrows.png")
            plt.savefig(p5, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {p5}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="M4: Pattern-Geometry — Wazn as Geometric Operator"
    )
    parser.add_argument('--output', default='m4_results',
                        help="Output directory (default: m4_results)")
    parser.add_argument('--model', choices=['camelbert', 'aravec'], default='camelbert',
                        help="Embedding model (default: camelbert)")
    parser.add_argument('--arabic',
                        default="C:/Users/ahmed/Desktop/Twitter AraVec/"
                                "full_grams_cbow_300_twitter/"
                                "full_grams_cbow_300_twitter.mdl",
                        help="Path to AraVec .mdl file (only needed with --model aravec)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Load Arabic model ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"M4 PATTERN GEOMETRY — {args.model.upper()}")
    print(f"{'='*60}")

    if args.model == 'camelbert':
        ar_tok, ar_mdl = load_bert(ARABIC_CLASSICAL_MODEL)
        print("\nEmbedding annotated Arabic lexicon (CAMeLBERT-ca, 3 carriers)...")
        table_ar = embed_annotated_lexicon(
            WORD_ANNOTATIONS, ar_tok, ar_mdl, ARABIC_CARRIERS,
            preprocess=strip_diacritics,
        )
        model_label = "CAMeLBERT-ca (contextual)"
        en_tok, en_mdl = load_bert("bert-base-uncased")
        print("\nEmbedding English lexicon (BERT-base, 3 carriers)...")
        table_en = embed_annotated_lexicon(
            ENGLISH_ANNOTATIONS, en_tok, en_mdl, ENGLISH_CARRIERS,
            preprocess=None,
        )
        en_model_label = "BERT-base-uncased (contextual)"
    else:
        wv = load_aravec(args.arabic)
        print("\nEmbedding annotated Arabic lexicon (AraVec static)...")
        table_ar = embed_aravec_lexicon(WORD_ANNOTATIONS, wv,
                                         preprocess=strip_diacritics)
        model_label = "AraVec Twitter (static)"
        table_en = []
        en_model_label = "N/A"

    # ── Geometric analysis ─────────────────────────────────────────────
    print("\nBuilding family-pattern map...")
    fpm_ar = get_family_pattern_map(table_ar)
    fpm_en = get_family_pattern_map(table_en) if table_en else {}

    print("Computing pattern offsets (anchor = V1)...")
    offsets_ar = compute_pattern_offsets(fpm_ar, anchor_pattern="V1")
    offsets_en = compute_pattern_offsets(fpm_en, anchor_pattern="V1") if fpm_en else {}

    print("Testing analogy consistency...")
    consistency_ar = analogy_consistency(offsets_ar)
    consistency_en = analogy_consistency(offsets_en) if offsets_en else {}

    print("Testing analogy completion (leave-one-out)...")
    completion_ar = analogy_completion(fpm_ar, offsets_ar, anchor_pattern="V1")
    completion_en = analogy_completion(fpm_en, offsets_en) if fpm_en else []

    print("Testing pattern clustering...")
    clustering_ar = pattern_clustering(table_ar)
    clustering_en = pattern_clustering(table_en) if table_en else {}

    # ── Print results ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("MEASUREMENT 4 RESULTS — ARABIC")
    print(f"Model: {model_label}")
    print(f"{'='*65}")

    print(f"\n── 1. ANALOGY CONSISTENCY (offset = vec(pattern) − vec(V1)) ──")
    print(f"   Higher = pattern is a stable geometric operator across roots")
    print(f"   {'Pattern':<8} {'n':>3}  {'mean_consistency':>18}  {'std':>8}  {'min':>8}  {'max':>8}")
    print(f"   {'-'*60}")
    for p in sorted(consistency_ar, key=lambda x: consistency_ar[x]['mean_consistency'], reverse=True):
        c = consistency_ar[p]
        flag = " ←" if p in PRIMARY_TRIAD else ""
        print(f"   {p:<8} {c['n_families']:>3}  {c['mean_consistency']:>18.4f}  "
              f"{c['std_consistency']:>8.4f}  {c['min']:>8.4f}  {c['max']:>8.4f}{flag}")

    print(f"\n── 2. ANALOGY COMPLETION (leave-one-out, cos(pred, actual)) ──")
    if completion_ar:
        by_p = {}
        for r in completion_ar:
            by_p.setdefault(r['pattern'], []).append(r['predicted_sim'])
        print(f"   {'Pattern':<8} {'n':>3}  {'mean_sim':>10}  {'std':>8}  {'interpretation'}")
        print(f"   {'-'*62}")
        for p in sorted(by_p, key=lambda x: np.mean(by_p[x]), reverse=True):
            m = np.mean(by_p[p])
            s = np.std(by_p[p])
            interp = ("strong" if m > 0.90 else "good" if m > 0.80
                      else "moderate" if m > 0.65 else "weak")
            print(f"   {p:<8} {len(by_p[p]):>3}  {m:>10.4f}  {s:>8.4f}  {interp}")

    print(f"\n── 3. PATTERN CLUSTERING (within vs cross-pattern similarity) ──")
    print(f"   {'Pattern':<8} {'n':>3}  {'within':>8}  {'cross':>8}  {'gap':>8}  {'sig':>5}")
    print(f"   {'-'*55}")
    for p in sorted(clustering_ar, key=lambda x: clustering_ar[x]['gap'], reverse=True):
        c = clustering_ar[p]
        sig = "YES" if c['significant'] else "no"
        print(f"   {p:<8} {c['n_words']:>3}  {c['within_mean']:>8.4f}  "
              f"{c['cross_mean']:>8.4f}  {c['gap']:>8.4f}  {sig:>5}")

    if consistency_en:
        print(f"\n── ENGLISH COMPARISON (BERT-base) ──")
        print(f"   {'Pattern':<8} {'n':>3}  {'mean_consistency':>18}")
        for p in ['V1', 'AP', 'PP', 'VN']:
            if p in consistency_en:
                c = consistency_en[p]
                print(f"   {p:<8} {c['n_families']:>3}  {c['mean_consistency']:>18.4f}")

    # ── Primary triad summary ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print("PRIMARY TRIAD SUMMARY: V1 (verb) → AP (فاعل) → PP (مفعول)")
    print(f"{'='*65}")
    for p in PRIMARY_TRIAD:
        if p in consistency_ar:
            c  = consistency_ar[p]
            cl = clustering_ar.get(p, {})
            comp_sims = [r['predicted_sim'] for r in completion_ar if r['pattern'] == p]
            print(f"\n  {p}:")
            print(f"    Offset consistency:    {c['mean_consistency']:.4f} ± {c['std_consistency']:.4f}  (n={c['n_families']} roots)")
            if comp_sims:
                print(f"    Completion accuracy:   {np.mean(comp_sims):.4f} ± {np.std(comp_sims):.4f}  (leave-one-out)")
            if cl:
                print(f"    Pattern clustering:    within={cl['within_mean']:.4f}  cross={cl['cross_mean']:.4f}  gap={cl['gap']:.4f}  sig={'YES' if cl['significant'] else 'no'}")

    # ── Visualizations ─────────────────────────────────────────────────
    print(f"\nGenerating visualizations → {args.output}/")
    generate_visualizations(table_ar, consistency_ar, completion_ar,
                             clustering_ar, table_en, consistency_en,
                             clustering_en, args.output)

    # ── Save JSON ──────────────────────────────────────────────────────
    def clean_table(table):
        return [{'family': r['family'], 'word': r['word'], 'raw': r['raw'],
                 'pattern': r['pattern'], 'description': r['description']}
                for r in table]

    output = {
        "experiment": "Measurement 4 — Pattern Geometry: Wazn as Geometric Operator",
        "framework": "AL-MIR'AH",
        "model": model_label,
        "annotated_words": len(table_ar),
        "arabic_analogy_consistency": consistency_ar,
        "arabic_analogy_completion": [
            {k: v for k, v in r.items()} for r in completion_ar
        ],
        "arabic_pattern_clustering": clustering_ar,
        "english_analogy_consistency": consistency_en,
        "english_pattern_clustering": clustering_en,
        "primary_triad_summary": {
            p: {
                "offset_consistency": consistency_ar.get(p, {}).get('mean_consistency'),
                "completion_mean": float(np.mean([r['predicted_sim'] for r in completion_ar
                                                   if r['pattern'] == p]))
                                   if any(r['pattern'] == p for r in completion_ar) else None,
                "pattern_gap": clustering_ar.get(p, {}).get('gap'),
            }
            for p in PRIMARY_TRIAD
        },
        "annotation_table": clean_table(table_ar),
    }

    path = os.path.join(args.output, "m4_results.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")

    print(f"\n{'='*65}")
    print("DONE")
    print(f"Results in: {args.output}/")
    print(f"  m4_results.json")
    print(f"  m4_pca_arabic.png          (pattern vs root coloring)")
    print(f"  m4_analogy_consistency.png (offset stability per pattern)")
    print(f"  m4_analogy_completion.png  (leave-one-out prediction)")
    print(f"  m4_pattern_clustering_gap.png")
    print(f"  m4_offset_arrows.png       (V1→AP and V1→PP per root)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
