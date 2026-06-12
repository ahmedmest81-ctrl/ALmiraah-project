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

- GitHub `engine/app.py`, `abjad.py`, `equilibrium.py`, `hyperbolic.py`, and
  `wazn.py` match the deployed Space versions byte-for-byte.
- The Abjad audit exists in GitHub and in the dataset repository.

### Blocking inconsistencies

- Public GitHub contains Paper B v4, not the current v5.
- The cited Hugging Face dataset contains 759 accumulated query records, not
  the 99-entry basis resource described in Paper B.
- The dataset repository has no `dataset_metadata.json`.
- The live Space contains `all_99_corrected.json` with 99 entries and 30
  fields. Paper B describes a 19-field public export that does not currently
  exist.
- The Space source lacks several fields named in Paper B's 19-field table,
  including corpus frequency, seasonal associations, color associations, and
  poetic meter. These cannot be generated faithfully without an additional
  source.
- GitHub README says the dataset has 747 records; the fetched dataset has 759.

## Required Before Public Submission

1. Publish Paper A v6 and Paper B v5 in place of the stale manuscript files.
2. Add the Paper A scripts, results, M4 review data, and this results manifest.
3. Decide Paper B's public dataset contract:
   - release a real 19-field basis export plus `dataset_metadata.json`, or
   - revise Paper B to describe the actual 30-field source and separately
     describe the 759-record query dataset.
4. Keep stored per-Name Abjad totals provisional until the five conventions
   are adjudicated.
5. Update README record counts and repository layout after the release files
   are finalized.
