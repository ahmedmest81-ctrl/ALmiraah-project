# AL-MIR'AH RAG Helper

`almiraah_rag.py` is a small retrieval layer for AL-MIR'AH. It does not call an
LLM. It retrieves relevant passages from a local Arabic corpus and attaches
diagnostics that are useful for interpretability work:

- token overlap
- heuristic root matches
- Abjad values for query terms
- nearest Abjad links between query tokens and passage tokens
- JSON output for downstream tools
- Markdown report for quick reading

## Example

Run from the project folder:

```powershell
python rag/almiraah_rag.py --corpus corpus.txt --query "<arabic query>" --top-k 5
```

For a faster test:

```powershell
python rag/almiraah_rag.py --corpus corpus.txt --query "<arabic query>" --top-k 3 --n-sentences 1000 --max-passages 600
```

Outputs are written to:

- `rag_results/almiraah_rag_results.json`
- `rag_results/almiraah_rag_report.md`

## Notes

If scikit-learn is installed, the helper uses TF-IDF retrieval. If not, it
falls back to a built-in BM25-style scorer, so the feature still works in a
minimal environment.

This is meant to be the evidence layer. A later LLM step can consume the JSON
and answer only from the returned passages.
