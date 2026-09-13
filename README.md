# AL-MIRʾĀH

[![CI](https://github.com/ahmedmest81-ctrl/ALmiraah-project/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmedmest81-ctrl/ALmiraah-project/actions/workflows/ci.yml)

AL-MIRʾĀH is a research prototype for inspecting classical Arabic terms against a fixed 99-Name relational basis. A CAMeLBERT-ca carrier-sentence embedding yields a signed attractor/repeller profile; the five strongest positive attractors place a query by a Karcher mean on a fitted Poincaré disk. The tools expose that profile, morphology, coordinates and pairwise geometry to an MCP client. Coordinates are **basis-relative measurements**, not definitions or direct English translations. Abjad fields are provisional.

The source of truth for project links and spelling is [CANONICAL.md](CANONICAL.md). The [Hugging Face Space](https://huggingface.co/spaces/AhmedMSLTI/almiraah_transformer) hosts the public service. Its MCP endpoint is `https://ahmedmslti-almiraah-transformer.hf.space/mcp`, and the [public query dataset](https://huggingface.co/datasets/AhmedMSLTI/almiraah_coordinate_db) holds accumulated term coordinates. [Paper A](https://doi.org/10.5281/zenodo.20735409) studies morphology and Abjad attention; [Paper B](https://doi.org/10.5281/zenodo.20739416) documents the fitted 99-Name coordinate system. [Portfolio](https://ahmedmislati.dev/).

## Run locally

Use Python 3.11, install once with `python -m pip install -r requirements.txt`, then run this single command from the repository root:

```sh
python engine/run_local.py
```

Open `http://127.0.0.1:7860/` for a simple lookup form or `/docs` for HTTP endpoints. First startup downloads CAMeLBERT-ca and may take several minutes. The launcher copies the released basis and disk fit into the engine working directory, and the server reads the public accumulated query dataset when available. No token is needed for read-only local use. Setting `HF_TOKEN` enables accumulation writes to the dataset **only if that token has write access**; leave it unset for inspection.

## Five MCP tools

Each response is inspectable text. The examples show abbreviated fields, not exact model scores. **Look up a term before comparing or projecting it** so a client can inspect what supports the interpretation.

| Tool | Example input | Abbreviated output |
| --- | --- | --- |
| `philological_lookup` | `{"term":"رحمة"}` | `PHILOLOGICAL COORDINATE: رحمة`; Abjad, top and bottom Names, `px/py/r`, tier and wazn |
| `root_analysis` | `{"root":"ر-ح-م"}` | `ROOT ANALYSIS: ر-ح-م`; matching basis Names, source fields, and cluster geometry when multiple Names share a root |
| `semantic_neighbors` | `{"term":"رحمة","k":5,"min_r":0.1,"max_r":0.95}` | `SEMANTIC NEIGHBORS: رحمة`; geodesic distance and coordinates from the accumulated query dataset |
| `compare_terms` | `{"term1":"رحمة","term2":"رأفة"}` | `COMPARISON: رحمة ↔ رأفة`; flat displacement, Poincaré geodesic, hierarchy load, midpoint, shared and divergent attractors |
| `semantic_project` | `{"candidates":{"mercy":["رحمة","رأفة"]},"context_arabic":["لطف"]}` | `SEMANTIC PROJECTION`; contextual candidate profiles and a best geometric fit when context exists |

`semantic_project` accepts **Arabic candidates supplied by the caller**. It does not translate English text by itself. Near the disk center, distinct terms can be compressed together; tier and radius can disagree. A geodesic neighbor is a lead for examination, not proof of synonymy.

## Repository structure and evidence

- `engine/`: FastAPI/MCP transport, the typed five-tool service, geometry, morphology, Abjad helpers, and local launcher.
- `data/paper_b/basis_99_v3.json`: the 99-entry, 30-field released basis; `poincare_data_v3.json`: its fitted disk positions. Field status in `dataset_metadata.json` distinguishes descriptive, provisional and framework-interpretive annotations.
- `data/paper_a/`, `experiments/`, `results/`: reviewed morphology annotations, statistical scripts and locked paper results.
- `pipeline/`: regeneration and validation of the v3 coordinate protocol.
- `tests/`: geometry, frequency-matched null and all-five-tool smoke tests.

The three data layers are the **99-Name basis**, the **accumulated Arabic query dataset**, and the **experimental annotations/results**. Their provenance and update rates differ; never treat one layer as another. [ARCHITECTURE.md](ARCHITECTURE.md) describes dependencies and the service boundary.

Run offline checks with `python -m unittest discover -s tests -v` and `python pipeline/validate_paper_b_dataset.py`. CI runs lint and unit tests without fetching a model or writing to Hugging Face. The Q2 frequency-matched permutation analysis is tested on deterministic fixtures; the published statistical result is a locked artifact, not recomputed in CI.

Paper A reports a weak Abjad-attention association and a morphology-general Hebrew control signal. It does **not** show a significant Hebrew gematria-specific increment over its permutation null. This code and its outputs are a methodological instrument, not a theological verdict or a claim that the model has learned Abjad numerology. See the papers for design, limits and falsification conditions.

Ahmed Mislati · MIT code license. The released basis and query data have their own licenses; see `data/paper_b/DATASET_LICENSE.md`.
