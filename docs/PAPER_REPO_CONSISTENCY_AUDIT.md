# Paper and Repository Consistency Audit

Audit date: 2026-06-12

## Fetched Sources

- GitHub `main`: commit `0a8af3b08fa7b48f0ba1186a4c746b43e804408e`
- Hugging Face dataset: commit `1aef46246bea66b64a460feec735ac001b1831e6`
- Hugging Face Space: commit `3dca0b6389dccd1b16b9e50c5c50d87beb8f0bdd`

The existing local checkout at `C:\Users\ahmed\ALmiraah-project` was fetched
without merging. Its dirty working tree was not changed.

## Paper A

### Correct in this release tree

- Final Paper A v6 includes the verified Hebrew sample definitions.
- The clustering `n = 209` is explicitly identified as word forms, not root
  families.
- Exact generating scripts and locked result JSONs are included.
- The reviewed M4 annotations, review log, and result JSON are included.

### Public GitHub issues found

- Public `papers/paper_A_v5.md` predates the verified denominator wording.
- Public GitHub omitted the Q2 primary script, unrestricted robustness script,
  Hebrew scripts, result JSONs, and reviewed M4 package.
- The public frequency-matched script did not match the locked run. It
  rank-transformed before residualization and filtered an obsolete
  `same_root` field. The release tree replaces it with the version that
  reproduces the saved result protocol.

## Paper B

### Consistent

- Before v3.2, GitHub `engine/app.py`, `abjad.py`, `equilibrium.py`,
  `hyperbolic.py`, and `wazn.py` matched the deployed Space versions
  byte-for-byte.
- The Abjad audit exists in GitHub and in the dataset repository.

### Resolved in this release tree

- Paper B v5 now describes the actual 99-entry, 30-field deployed basis source.
- The exact source and fitted v3 disk are released under `data/paper_b/`.
- `dataset_metadata.json` documents all fields, status categories, source
  commits, caveats, and checksums.
- The manuscript now distinguishes the fixed basis from the 759-record
  accumulated query dataset on Hugging Face.
- Fields absent from the source (corpus frequency, seasonal associations,
  color associations, and poetic meter) are no longer claimed.
- At audit time, the public GitHub README said 747 records; this release tree
  corrects the count to the fetched dataset's 759 records.

### Remaining qualification

- The v3.2 `engine/app.py` and `engine/hyperbolic.py` changes must be deployed
  to the Hugging Face Space together. Until then, the branch is ahead of the
  live tool and byte-for-byte parity is intentionally broken.
- Stored per-Name Abjad totals remain provisional until the five conventions
  are adjudicated. The paper and metadata label them accordingly.
- `paired_opposite` is a mixed free-text field rather than a normalized,
  uniformly attested opposition registry; the paper and metadata now state
  this limitation.
