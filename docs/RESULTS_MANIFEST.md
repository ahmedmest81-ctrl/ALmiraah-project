# Results Manifest

This manifest maps the principal empirical claims in Papers A and B to the
public or release-staged artifacts that support them.

## Paper A

| Claim | Script | Result |
| --- | --- | --- |
| Arabic Abjad-attention primary analysis | `experiments/q2_abjad_attention.py` | `results/paper_a/q2_results_with_robustness.json` |
| Unrestricted permutation and random-pair controls | `experiments/q2_robustness.py` | `results/paper_a/q2_results_with_robustness.json` |
| Frequency-matched permutation null | `experiments/q2_robustness_freq_matched.py` | `results/paper_a/q2_freq_matched_K4.json` |
| Hebrew AlephBERT control | `experiments/q3_alephbert_control.py` | `results/paper_a/q3_results_with_robustness.json` |
| Hebrew permutation and random-pair controls | `experiments/q3_robustness.py` | `results/paper_a/q3_results_with_robustness.json` |
| Reviewed 200-family Wazn run | `experiments/m4_pattern_geometry.py`, `experiments/m4_pattern_geometry_200.py` | `results/paper_a/m4_results_contextual_200_reviewed.json` |
| Wazn annotation review | review procedure recorded in the log | `data/paper_a/m4_annotations_200_reviewed.json`, `data/paper_a/m4_annotations_200_review_log.json` |

`q1_root_cluster_density.py` supplies the shared Arabic diacritic-stripping
utility imported by the M4 scripts. Its historical Q1 density results are
background analyses, not inputs to the locked 200-family result.

### Locked result checksums

| File | SHA-256 |
| --- | --- |
| `q2_results_with_robustness.json` | `2E4E31E15A9E7F3A8A09A8EA5DE0B9A4324F606C934F85947873B6FE06022FB8` |
| `q2_freq_matched_K4.json` | `8A55A42FC51D1AD501465F2687EEFDA2BFA60A775EC7FD70113566D7895BDF66` |
| `q3_results_with_robustness.json` | `F5CD06A88D0CE1F5FFAAD145AECFEFE14B7A542BF2850DC8290A5CAED2D35D95` |
| `m4_results_contextual_200_reviewed.json` | `EFD3F87EB7B852BB822D860095BCDB7CABAC2175D9B45CD25F0AF9C4AAB8D440` |

### Verified denominators

- Arabic primary: 2,322,977 confidently cross-root pairs.
- Hebrew primary: 12,081,207 all analyzable pairs.
- Hebrew confidently different-root subset: 1,468,657 pairs.
- Hebrew same-root subset: 58,776 pairs.
- Wazn reviewed lexicon: 951 forms across 200 unique triliteral families.
- Verbal-noun clustering: 209 forms across 200 families; nine families
  contribute two verbal-noun forms.

## Paper B

The GitHub engine modules match the files currently deployed in the Hugging
Face Space at the pre-v3.2 audited commit. The v3.2 engine changes in this
branch require a synchronized Space deployment. The exact deployed basis
source and fitted disk are released in this tree:

- `data/paper_b/basis_99_v3.json`
- `data/paper_b/poincare_data_v3.json`
- `data/paper_b/dataset_metadata.json`
- `pipeline/validate_paper_b_dataset.py`

The basis has 99 entries and all 30 source fields are present in every entry.
The metadata separates engine-facing, provisional Abjad, documentary, and
framework-interpretive fields. The separately hosted Hugging Face dataset is
the accumulated query dataset (759 records at audit), not the fixed basis.

### Released checksums

| File | SHA-256 |
| --- | --- |
| `basis_99_v3.json` | `08A510B63A292545C1C176C9C999AAAB0D76EE78D490D76B8EAA8952B2C32E2C` |
| `poincare_data_v3.json` | `8AA70364F07EA5E19B3321BEA38A5DE04919C95A55D0A7014C920C4DF77537F9` |
| `v32_geometry_diagnostics.json` | `58B21E7A720CF58ECB60666FFA08F2B0FADE173B9EA6E4B32CCD303F97687460` |
| `baseline_cosines_v3.json` | `87403189ACC56B32CB963E9C205A7DCD4B7EB336FB0802A488E9B571F1842E8B` |

### V3.2 diagnostics

| Claim | Script | Result |
| --- | --- | --- |
| Raw CAMeLBERT-ca cosine baseline values for the four Paper B demonstration pairs | `experiments/paper_b_baseline_cosines.py` | `results/paper_b/baseline_cosines_v3.json` |
| Geodesic midpoint and radial/angular diagnostics for the four Paper B pairs and exploratory *fanāʾ/baqāʾ* | `experiments/v32_geometry_diagnostics.py` | `results/paper_b/v32_geometry_diagnostics.json` |
