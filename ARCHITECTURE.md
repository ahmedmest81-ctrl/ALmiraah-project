# AL-MIRʾĀH architecture

Identity strings and public links come only from [CANONICAL.md](CANONICAL.md). This document explains local code and data boundaries; it does not define new public endpoints.

## Request path

1. `engine/app.py` starts CAMeLBERT-ca and loads the released 99-Name basis plus fitted Poincaré positions. FastAPI routes serve the local browser/JSON API; the MCP Streamable HTTP session manager serves `/mcp`.
2. The v3 protocol inserts an undiacritized Arabic term into three fixed carrier sentences, pools target token spans from layer 8, and averages them. Centered cosine similarities against the 99 basis vectors produce the signed profile. Top-five positive weights place the query with a disk-native Karcher mean. Tier is voted from attractors, not merely read from radius.
3. `engine/tool_service.py` is a model-independent application service. Its `SemanticBackend` protocol and `LiveBackend` adapter describe the query, profile, geometry, dataset and morphology operations required by all five tools. The MCP callback only converts service text to a `TextContent` block. A fake backend tests tool order, inputs and responses without downloading a model.
4. `engine/hyperbolic.py` owns disk-native distance, midpoint, barycenter and cluster mathematics. Flat Euclidean displacement and Poincaré geodesic are reported separately; Euclidean averaging is not used for placement.
5. `engine/wazn.py`, `abjad.py` and `equilibrium.py` contain specialized analyses. Wazn parses can be ambiguous; basis fields have distinct evidential status.

The code remains a research server. `app.py` still performs model startup and query persistence, while tool orchestration lives in a typed, testable service. A future deployment refactor can inject model and storage clients into a longer-lived engine object without changing MCP tool contracts.

## Three data layers

| Layer | Files / location | Role | Guardrail |
| --- | --- | --- | --- |
| Fixed 99-Name basis | `data/paper_b/basis_99_v3.json`, `poincare_data_v3.json`, `dataset_metadata.json` | Source annotations and fitted reference geometry | Preserve field-level evidential labels; do not present interpretive fields as sourced facts. |
| Accumulated query coordinates | Public dataset identified in `CANONICAL.md`, loaded from `coordinates.jsonl` when available | Previously queried Arabic terms for neighbor search | Version separately from basis, filter calibration noise, avoid protocol mixing. |
| Experiment annotations and results | `data/paper_a/`, `experiments/`, `results/` | Reviewed wazn annotations, frequency-matched null, locked outputs | Research evidence, not an extension of the live coordinate database. |

The local launcher copies released basis files into the engine directory; generated copies are ignored by Git. It may fetch cached basis vectors, while a first run without them embeds the 99 Names. The public query dataset is read without a token if no local copy exists. `HF_TOKEN` authorizes persistence and should remain unset for read-only inspection.

## Tool dependency order

`philological_lookup(term)` is the grounding primitive. `root_analysis(root)` inspects fixed-basis entries sharing a root. `semantic_neighbors(term)` uses accumulated query coordinates; it calls lookup first when a coordinate is not already saved. `compare_terms(term1, term2)` **looks up both terms before** computing flat and geodesic distances, midpoint, decomposition and profile overlap. `semantic_project(candidates, context_arabic)` looks up each supplied Arabic candidate and context term before contextual fit. Service tests assert comparison and unknown-neighbor lookup order.

The five MCP names and arguments are in [README.md](README.md). HTTP routes expose related JSON operations but are a separate transport surface. The repository contains no direct English-to-Arabic translation model; a future English bot must supply and verify candidate senses before projection.

## Validation and deployment boundary

CI lints the new application service and tests, compiles the engine, and runs offline unit tests. The Q2 test checks orthographic normalization, unique-word frequencies, within-tier permutation invariants, cross-root selection and empirical-p calculation on a fixture. It does not rerun the long published experiment. The existing geometry suite tests intrinsic disk operations.

`engine/Dockerfile` builds from the repository root with `docker build -f engine/Dockerfile .`; it copies released basis and engine code. The Hugging Face Space is a separate deployment target and may use a different commit. Changes here are not live until published there.
