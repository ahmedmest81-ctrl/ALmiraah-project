# AL-MIRAAH — Arabic NLP Grounding via MCP

AL-MIRAAH is a deployed research prototype that uses classical Arabic
morphology, a fixed 99-Name semantic basis, and the mathematical pipeline
documented in the accompanying research to produce structured semantic
grounding profiles. Its five MCP tools let a connected AI system reason with
inspectable evidence—attractors, repelled Names, morphology, coordinates,
distances, and confidence signals—instead of inventing an interpretation
without a reference frame.

## What this repository demonstrates

- Designing clear tool contracts for an AI agent
- Deploying an MCP service backed by a public, versioned dataset
- Translating a specialist research question into a usable interface
- Reproducible statistical evaluation with explicit falsification conditions
- Reporting limitations and negative results alongside positive findings

**Live interface:** https://ahmedmslti-almiraah-transformer.hf.space/

**MCP endpoint:** https://ahmedmslti-almiraah-transformer.hf.space/mcp

**Hugging Face Space:** https://huggingface.co/spaces/AhmedMSLTI/almiraah_transformer

## How an Arabic term is grounded mathematically

For an Arabic term $t$, the deployed v3 pipeline follows four documented steps.

1. **Contextual embedding.** The undiacritized term is inserted into three fixed
   Arabic carrier sentences. Its layer-8 target-token spans are mean-pooled and
   then averaged across carriers:

   $$
   e(t)=\frac{1}{3}\sum_{c=1}^{3}\frac{1}{|S_{t,c}|}
   \sum_{i\in S_{t,c}}h_{i,c}^{(8)}.
   $$

2. **Centered similarity to the basis.** For basis Name $n_j$, similarity is the
   cosine of vectors centered by the mean basis embedding $\mu_B$:

   $$
   s_j(t)=\frac{(e(t)-\mu_B)\cdot(e(n_j)-\mu_B)}
   {\|e(t)-\mu_B\|\,\|e(n_j)-\mu_B\|}.
   $$

   The tools also report an equilibrium-adjusted score $s_j(t)-\mu_j$, where
   $\mu_j$ is the average pull received by Name $j$ across the reference field.

3. **Hyperbolic placement.** The five strongest attractors receive non-negative
   weights $w_j=\max(0,s_j)$. The term's disk position is their weighted Karcher
   mean:

   $$
   p(t)=\operatorname*{arg\,min}_{p\in\mathbb{D}}
   \sum_{j\in N_5(t)}\hat w_j\,d_{\mathbb{D}}(p,p_j)^2,
   \qquad \hat w_j=\frac{w_j}{\sum_k w_k}.
   $$

   The implementation solves this intrinsic barycenter iteratively with
   $x\leftarrow\exp_x\!\left(\sum_j\hat w_j\log_x(p_j)\right)$.

4. **Geodesic comparison.** Distances on the Poincaré disk use

   $$
   d_{\mathbb{D}}(u,v)=\operatorname{arcosh}\!\left(
   1+\frac{2\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right).
   $$

These calculations produce a semantic grounding profile against this specific
basis; they do **not** automatically translate a word or establish one final,
model-independent meaning. Morphological parses can be ambiguous, Abjad values
are explicitly provisional, and confidence falls as geodesic distance to the
primary attractor increases.

The v3.2 branch must be deployed to the Space with `engine/app.py` and
`engine/hyperbolic.py` together. Until then, the live tool remains on the
previously audited engine version.

**Code:** https://github.com/ahmedmest81-ctrl/ALmiraah-project
(MCP server — five tools: `philological_lookup`, `root_analysis`,
`semantic_neighbors`, `compare_terms`, `semantic_project`)

**Basis dataset:** `data/paper_b/basis_99_v3.json`
(99 Names × 30 fields, with schema/status metadata and fitted coordinates)

**Query dataset:** https://huggingface.co/datasets/AhmedMSLTI/almiraah_coordinate_db
(832 accumulated query records at the current public dataset commit, v3 protocol)

**Papers**
- *Paper A* — [Templatic Morphology as Decodable Geometry, and Abjad
  Letter-Values as an Attention Probe](https://doi.org/10.5281/zenodo.20735409)
- *Paper B* — [The 99-Names Coordinate System: A Fitted Relational Basis for
  Classical Arabic in CAMeLBERT-ca](https://doi.org/10.5281/zenodo.20739416)

## Research origin

I work inside classical Arabic, and I'd long had the intuition that its
root-pattern morphology behaves like a coordinate system, that form and meaning
are mathematically coupled in a way most languages don't make legible. This
repository is the attempt to find out whether that intuition was real or just
felt real: if the structure is there, a transformer trained on Arabic should
show it, and the 8th-century Mashriqi Abjad encoding should leave a measurable
trace. What follows is the experiment that demanded.

The framework maps classical Arabic vocabulary onto a fitted Poincaré disk using
the 99 Names of God (al-Asmāʾ al-Ḥusnā) as a fixed semantic basis. It produces
relational coordinates from CAMeLBERT-ca embeddings and tests whether Arabic's
root-pattern morphology (wazn) and the Mashriqi Abjad numeral system produce
measurable geometric structure in transformer representations.

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
                    Euclidean/hyperbolic distances with hierarchy load;
                    v3.2: geodesic midpoint + radial/angular diagnostic
                    in compare_terms, root-cluster intrinsic geometry
                    in root_analysis)
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
  validate_paper_b_dataset.py  Validate the released 99×30 basis contract

experiments/   Pre-registered analyses with honest tallies
  q2_abjad_attention.py             Paper A primary Abjad-attention analysis
  q2_robustness.py                  Paper A unrestricted Q2 robustness analysis
  q2_robustness_freq_matched.py   Paper A frequency-matched null
  q3_alephbert_control.py            Paper A AlephBERT control
  q3_robustness.py                   Paper A Q3 robustness analysis
  m4_pattern_geometry.py             Wazn-pattern geometry analysis
  m4_pattern_geometry_200.py         Locked 200-family Wazn run
  v32_geometry_diagnostics.py        Saved-query midpoint/decomposition audit
  three_experiments_report.md     Opposition axes / shadow field / wazn×position
  abjad_audit_report.txt          99-Name stored-value audit
  opposition_edges.json           Machine-resolved doctrinal opposition graph
  color_profiles.json             99-dim profiles of the five color terms

docs/          DEPLOYMENT.md (fix history), VALIDATION_V3.md (protocol
               correction results), paper consistency audit, and result manifest

data/paper_a/  Reviewed 200-family Wazn annotations and adjudication log

data/paper_b/  Exact deployed 99-entry basis source, fitted v3 Poincaré
               coordinates, machine-readable metadata, and dataset licence

results/paper_a/ Locked result JSONs cited by Paper A

results/paper_b/ Locked v3.2 geometry diagnostics cited by Paper B/C
```

## Reproduction

```bash
pip install -r requirements.txt
python pipeline/regenerate_v3.py            # re-embed basis, re-fit disk
python pipeline/regenerate_full_v3.py       # re-project accumulated terms
python pipeline/validate_paper_b_dataset.py # verify 99×30 schema and hashes
python -m unittest tests/test_hyperbolic_operations.py -v
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
