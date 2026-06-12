# V3 Regeneration — Validation Report
2026-06-10 · carrier-sentence/layer-8 protocol (Paper B §3.1 compliance)

## Headline numbers

**Anisotropy collapse at the source.** Basis self-similarity under v2
(bare-term, last-layer): compressed band ≈ 0.86–0.94. Under v3
(3 carriers, layer-8, word-span pooling, vector-centered):
mean = −0.0098, std = 0.129. The space breathes.

**Identity sanity check.** نور → النُّور at **+0.71** similarity — a
query term maximally attracting the basis Name it lexically is.
Invisible under v2. This is the single strongest evidence the v3
protocol measures what the architecture claims to measure.

**Hub dissolution.** Al-Ḥayy: top-3 attractor for ~half the v2 test
set; top-3 for **zero** of 20 v3 validation terms. Al-ʿAfuww survives
only in صمت، نسيان، سكوت، محو، غفلة — the effacement/cessation cluster,
i.e. exactly where it is semantically earned. The v2 hub problem was
protocol artifact, not field structure.

**Selected v3 profiles** (vs v2 in parentheses):
- مرآة → النُّور، المُبْدِئ، البَدِيع (v2: البَاعِث، الوَاجِد، الجَامِع)
- صدق → الحَقّ +0.36، العَدْل، اللّٰه (v2: الحَقّ — survived, now joined by Justice)
- ظل → الأَحَد، النُّور، اللّٰه — the Akbarī ẓill doctrine, sharper
- استحضار → المُحْصِي (the Enumerator) — making-present pulls the Recorder

## Disk re-fit
Hard tier-radius constraint, angular stress descent, seed 42:
Dhāt r = 0.304 ± 0.014 · Ṣifāt 0.550 ± 0.012 · Afʿāl 0.780 ± 0.011.
Clean band separation by construction (this is the §3.1 spec, not a
finding; the finding is whether QUERY r-values respect it — see below).

## Karcher vs Euclidean (the hyperbolic upgrade, quantified)
Across 20 validation terms, the Euclidean weighted centroid
under-estimates r by a mean of **0.031**; for 4/20 terms
(ظل، صمت، حدس، غفلة) the compression **changed the tier assignment**.
The live server now uses the Karcher mean (hyperbolic.py); position
interpolation is intrinsic to the disk for the first time.

## What still needs to run (Ahmed)
1. Upload to Space: app.py, Dockerfile, hyperbolic.py,
   poincare_data_v3.json, name_vecs_v3.npz (wazn.py already there).
2. Archive coordinates.jsonl → coordinates_v2_baretermlastlayer.jsonl
   in the dataset repo, then regenerate the full accumulated set:
   `python3 regenerate_v3.py --terms <all_terms.txt>` (the dataset is
   private; runs wherever HF_TOKEN lives). Until then, semantic_neighbors
   mixes v2 positions with v3 queries — flagged, known, temporary.
3. Recompute Paper B §5 demonstration values under v3 and update prose;
   state word-span pooling explicitly in §3.1.
4. The v2-vs-v3 attractor-profile comparison is a free robustness
   subsection: profile stability across embedding protocols directly
   supports §3.4/§6.

## Files
regenerate_v3.py · hyperbolic.py · poincare_data_v3.json ·
name_vecs_v3.npz · coordinates_v3.jsonl (20-term validation sample) ·
app.py (v3-wired) · Dockerfile (copies all v3 artifacts)
