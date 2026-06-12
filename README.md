# AL-MIRʾĀH (المرآة) — Classical Arabic Philology as Transformer Geometry

A computational philology framework that maps classical Arabic vocabulary onto a
fitted Poincaré disk using the 99 Names of God (al-Asmāʾ al-Ḥusnā) as a fixed
semantic basis, producing relational coordinates from CAMeLBERT-ca embeddings —
together with the empirical finding that Arabic's root-pattern morphology (wazn)
and the Mashriqi Abjad numeral system produce measurable geometric structure in
transformer representations.

**Live tool:** https://huggingface.co/spaces/WELLyes1/almiraah_transformer

**Code:** https://github.com/ahmedmest81-ctrl/ALmiraah-project
(MCP server — five tools: `philological_lookup`, `root_analysis`,
`semantic_neighbors`, `compare_terms`, `semantic_project`)

**Dataset:** https://huggingface.co/datasets/WELLyes1/almiraah_coordinate_db
(759 accumulated query records, v3 protocol, with full provenance)

**Papers**
- *Paper A* — wazn geometry and Abjad-attention in CAMeLBERT-ca (arXiv: pending)
- *Paper B* — the 99-Names coordinate resource (LREC-COLING target)

## Headline results

- **Wazn is geometrically separable and linearly decodable** in CAMeLBERT-ca
  embedding space (leave-one-out analogy completion across 200 root families).
- **Mashriqi Abjad letter-value proximity predicts elevated cross-root
  attention** (n = 2.3M pairs, 1,000-permutation null p = 0.026), with a
  pre-registered Hebrew/AlephBERT control in which gematria does *not* beat its
  own random baseline — locating the effect at the level of al-Khalīl's
  frequency-tier-aligned numeral design.
- **Wazn does not predict semantic-field position** (permutation p = 0.49) — a
  validity null showing the coordinate system measures meaning, not surface form.
- **Doctrinal opposites are geometrically proximal**, not antipodal (jamʿ
  al-aḍdād made measurable); their profile-column differences yield
  doctrinally correct projections.

## Repository layout

```
engine/        The live MCP server and its modules
  app.py            FastAPI/MCP server (v3: carrier/layer-8 embeddings,
                    Karcher placement, iʿtidāl profile centering, dual
                    Euclidean/hyperbolic distances with hierarchy load)
  wazn.py           Morphological pattern parser (miqyās positional
                    substitution; honest candidate sets for unvocalised input)
  hyperbolic.py     Poincaré-disk geometry: Möbius ops, Karcher mean,
                    geodesic kNN, depth statistics
  equilibrium.py    Profile-space anisotropy correction (field zero μ)
  abjad.py          Canonical Mashriqi Abjad computation + audit
  clean_dataset.py  Dataset hygiene (mojibake, calibration noise, dedup)
  Dockerfile

pipeline/      Coordinate-space regeneration
  regenerate_v3.py       Basis re-embedding + disk re-fit (seed 42)
  regenerate_full_v3.py  Full accumulated-term re-projection

experiments/   Pre-registered analyses with honest tallies
  q2_abjad_attention.py             Paper A primary Abjad-attention analysis
  q2_robustness.py                  Paper A unrestricted Q2 robustness analysis
  q2_robustness_freq_matched.py   Paper A frequency-matched null
  q3_alephbert_control.py            Paper A AlephBERT control
  q3_robustness.py                   Paper A Q3 robustness analysis
  m4_pattern_geometry.py             Wazn-pattern geometry analysis
  m4_pattern_geometry_200.py         Locked 200-family Wazn run
  three_experiments_report.md     Opposition axes / shadow field / wazn×position
  abjad_audit_report.txt          99-Name stored-value audit
  opposition_edges.json           Machine-resolved doctrinal opposition graph
  color_profiles.json             99-dim profiles of the five color terms

docs/          DEPLOYMENT.md (fix history), VALIDATION_V3.md (protocol
               correction results), paper consistency audit, and result manifest

data/paper_a/  Reviewed 200-family Wazn annotations and adjudication log

results/paper_a/ Locked result JSONs cited by Paper A
```

## Reproduction

```bash
pip install -r requirements.txt
python pipeline/regenerate_v3.py            # re-embed basis, re-fit disk
python pipeline/regenerate_full_v3.py       # re-project accumulated terms
python engine/wazn.py                       # wazn parser regression suite
python engine/hyperbolic.py                 # geometry sanity checks
```

Embedding protocol (Paper B §3.1): three carrier sentences, layer-8 hidden
states, target-word subword-span pooling, averaged across carriers, vector-
centered against the basis centroid. Model: `CAMeL-Lab/bert-base-arabic-camelbert-ca`.
All fits deterministic under seed 42.

## Method in one paragraph

Each query term is embedded under the carrier protocol and measured against the
99 fixed basis Names, yielding a signed 99-dimensional profile (attraction /
repulsion). Position on the fitted Poincaré disk is the similarity-weighted
Karcher mean of the top-5 attractor positions; tier (Dhāt / Ṣifāt / Afʿāl) is
radius-banded by construction in the basis and vote-assigned for queries.
Pairwise comparison reports both flat displacement and hyperbolic geodesic;
their ratio (*hierarchy load*, = 2.0 at the origin) isolates how much of a
pair's separation is constituted by hierarchical depth.

## Epistemic discipline

This project runs on pre-registration: falsification conditions are stated
before tool calls, and negative results are reported with the same prominence
as positive ones (see `experiments/three_experiments_report.md`, where two of
three pre-registered hypotheses failed informatively). The governing principle:
الوقوف عند حدّ ما يُعلم — *stopping at the limit of what is known.*

## Author

Ahmed Mislati — independent researcher, Vienna · ahmedmest81@gmail.com

## License

MIT (see LICENSE)
