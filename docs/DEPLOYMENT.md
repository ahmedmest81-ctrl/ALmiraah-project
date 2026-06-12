# AL-MIR'ĀH — Fix Package & Deployment Checklist
2026-06-10 · companion to the live tool test of the same date

## Contents
| File | Fixes | Server touch-point |
|---|---|---|
| `wazn.py` | Wazn inheritance bug | `philological_lookup`, `semantic_project` |
| `equilibrium.py` | Hub-attractor / anisotropy | all five tools' attractor ranking |
| `clean_dataset.py` | Mojibake, calibration noise, dupes | `semantic_neighbors`, `semantic_project` neighbor lists |
| `abjad.py` + `abjad_audit_report.txt` | Header/breakdown divergence + stored-value audit | `philological_lookup`, `root_analysis` |

## Order of operations (each step independently shippable)

**1. Wazn fix — ship first (papers cite the field).**
   - Add `wazn.py` to the Space repo.
   - In `philological_lookup`, replace the `dominant_wazn` assignment:
     `record.update(wazn.lookup_output_fields(term, cluster_majority_wazn))`
   - Output now carries `query_wazn` + `query_wazn_status` + `cluster_wazn`
     as three separate fields. Remove or deprecate `dominant_wazn`
     (if kept for API compatibility, alias it to `query_wazn`, never
     to cluster wazn).
   - Re-run the six Paper B §5 demonstration terms and the
     almiraah_six_layers terms; update any wazn mention in Papers A/B.
   - Regression: `python3 wazn.py` must show all bug-case terms
     returning parsed/candidate output, never a cluster pattern.

**2. Dataset cleaning — ship second (cheap, enables step 3).**
   - `python3 clean_dataset.py 1779798638098_coordinates__4_.jsonl`
   - Review the `.removed.jsonl` audit (expected: number-words,
     ≥1 mojibake repair, possible dupes), then point the server's
     neighbor index at the `.cleaned.jsonl`.

**3. Iʿtidāl correction — ship third (needs the cleaned set).**
   - Compute μ once from the cleaned query set
     (`field_zero_from_queries`); store beside the basis as
     `field_zero.npy`. It is part of the Anchor Database: frozen,
     recomputed only on deliberate revision.
   - Expose BOTH rankings (`attractors_raw`, `attractors_centered`)
     for one validation cycle. The accumulated dataset and Paper B §5
     values were computed on raw profiles — do not silently change
     published numbers.
   - Run `hub_concentration` before/after on the cleaned set; the
     before/after table is a ready-made Paper B robustness appendix
     (and an honest strengthening: it documents a limitation and its
     principled correction).

**4. Abjad — adjudicate before shipping.**
   - `abjad.py` guarantees header = breakdown by construction; safe to
     ship for display formatting immediately (`format_line`).
   - The 15 flagged stored values in `abjad_audit_report.txt` need
     your ruling on five conventions before any data correction:
     (a) article-free vs article-inclusive canonical value;
     (b) madda آ = 1 or 2; (c) hamza-seat ؤ/ئ valuation;
     (d) tāʾ marbūṭah = 400 (current) — confirm;
     (e) multi-word Names (ذو الجلال والإكرام) — full or truncated form.
   - After ruling: regenerate `abjad_num` for all 99 via `abjad.py`,
     bump all_99_corrected.json version, re-check Paper B tables.
     Paper A's Q2 pipeline used per-letter values and is unaffected
     in principle — verify §3 examples.

## What was NOT touched
The 99-Name basis, the Poincaré fit, the carrier sentences, and all
published §5 values. Every fix sits at the measurement/hygiene layer,
consistent with the Anchor Database discipline: the reference is
constant; the instrument reading it gets corrected.
