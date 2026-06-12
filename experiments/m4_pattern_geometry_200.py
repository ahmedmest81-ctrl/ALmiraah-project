"""
Same-standard M4 runner for the 200-family Arabic wazn expansion.

This script keeps the original M4 analysis functions, Arabic carrier
sentences, preprocessing, leave-one-out completion, clustering, and JSON
schema. The only changes are:

  * annotations are loaded from m4_annotations_200.json
  * CAMeLBERT/BERT models run on CUDA when available
  * output is written to a separate directory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import m4_pattern_geometry as base  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def load_annotations(path: Path, limit_families: int | None = None):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["annotations"]
    if limit_families:
        keep = []
        seen = []
        for row in rows:
            fam = row["family"]
            if fam not in seen:
                if len(seen) >= limit_families:
                    continue
                seen.append(fam)
            keep.append(row)
        rows = keep
    return [
        (row["family"], row["raw"], row["pattern"], row["description"])
        for row in rows
    ], data


def load_bert_on_device(model_name: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    print(f"  Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def get_contextual_vector(tokenizer, model, word: str, carrier: str, device: str):
    inputs = tokenizer(carrier, return_tensors="pt", truncation=True, max_length=64)
    word_tokens = tokenizer.tokenize(word)
    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    match = None
    for i in range(len(all_tokens) - len(word_tokens) + 1):
        if all_tokens[i : i + len(word_tokens)] == word_tokens:
            match = list(range(i, i + len(word_tokens)))
            break
    if match is None:
        return None

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[0][match].mean(dim=0).detach().cpu().numpy()


def get_averaged_vector(tokenizer, model, word: str, carriers, device: str):
    vecs = []
    for template in carriers:
        vec = get_contextual_vector(tokenizer, model, word, template.format(word), device)
        if vec is not None:
            vecs.append(vec)
    return np.mean(vecs, axis=0) if vecs else None


def embed_annotated_lexicon(annotations, tokenizer, model, carriers, device: str, preprocess=None):
    table = []
    missing = []
    seen = set()
    total = len(annotations)
    for idx, (family, raw_word, pattern, desc) in enumerate(annotations, start=1):
        word = preprocess(raw_word) if preprocess else raw_word
        key = (family, word, pattern)
        if key in seen:
            continue
        seen.add(key)
        vec = get_averaged_vector(tokenizer, model, word, carriers, device)
        if vec is not None:
            table.append(
                {
                    "family": family,
                    "word": word,
                    "raw": raw_word,
                    "pattern": pattern,
                    "description": desc,
                    "vector": vec,
                }
            )
        else:
            missing.append({"family": family, "word": word, "pattern": pattern})
        if idx % 100 == 0:
            print(f"  Embedded progress: {idx}/{total} annotations")
    print(
        f"  Embedded {len(table)} words | missing: {len(missing)}"
        + (f" {missing[:5]}" if missing else "")
    )
    return table, missing


def clean_table(table):
    return [
        {
            "family": r["family"],
            "word": r["word"],
            "raw": r["raw"],
            "pattern": r["pattern"],
            "description": r["description"],
        }
        for r in table
    ]


def summarize_completion(completion_ar):
    by_p = {}
    for row in completion_ar:
        by_p.setdefault(row["pattern"], []).append(row["predicted_sim"])
    return {
        pattern: {
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for pattern, values in sorted(by_p.items())
    }


def run(args):
    os.makedirs(args.output, exist_ok=True)
    annotations, annotation_meta = load_annotations(args.annotations, args.limit_families)
    families = sorted({row[0] for row in annotations})

    print(f"\n{'=' * 60}")
    print("M4 PATTERN GEOMETRY — 200-FAMILY SAME-STANDARD RUN")
    print(f"{'=' * 60}")
    print(f"Arabic families requested: {len(families)}")
    print(f"Arabic annotations requested: {len(annotations)}")

    ar_tok, ar_mdl = load_bert_on_device(base.ARABIC_CLASSICAL_MODEL, args.device)
    print("\nEmbedding annotated Arabic lexicon (CAMeLBERT-ca, 3 carriers)...")
    table_ar, missing_ar = embed_annotated_lexicon(
        annotations,
        ar_tok,
        ar_mdl,
        base.ARABIC_CARRIERS,
        args.device,
        preprocess=base.strip_diacritics,
    )
    model_label = "CAMeLBERT-ca (contextual, same-standard 200-family expansion)"

    table_en = []
    consistency_en = {}
    clustering_en = {}
    en_model_label = "not run"
    if not args.skip_english:
        en_tok, en_mdl = load_bert_on_device("bert-base-uncased", args.device)
        print("\nEmbedding English lexicon (BERT-base, 3 carriers)...")
        table_en, _ = embed_annotated_lexicon(
            base.ENGLISH_ANNOTATIONS,
            en_tok,
            en_mdl,
            base.ENGLISH_CARRIERS,
            args.device,
            preprocess=None,
        )
        en_model_label = "BERT-base-uncased (contextual)"

    print("\nBuilding family-pattern map...")
    fpm_ar = base.get_family_pattern_map(table_ar)
    fpm_en = base.get_family_pattern_map(table_en) if table_en else {}

    print("Computing pattern offsets (anchor = V1)...")
    offsets_ar = base.compute_pattern_offsets(fpm_ar, anchor_pattern="V1")
    offsets_en = base.compute_pattern_offsets(fpm_en, anchor_pattern="V1") if fpm_en else {}

    print("Testing analogy consistency...")
    consistency_ar = base.analogy_consistency(offsets_ar)
    consistency_en = base.analogy_consistency(offsets_en) if offsets_en else {}

    print("Testing analogy completion (leave-one-out)...")
    completion_ar = base.analogy_completion(fpm_ar, offsets_ar, anchor_pattern="V1")
    completion_en = base.analogy_completion(fpm_en, offsets_en) if fpm_en else []

    print("Testing pattern clustering...")
    clustering_ar = base.pattern_clustering(table_ar)
    clustering_en = base.pattern_clustering(table_en) if table_en else {}

    print("\nPrimary completion summary:")
    for pattern, stats in summarize_completion(completion_ar).items():
        print(
            f"  {pattern:<8} n={stats['n']:>3} "
            f"mean={stats['mean']:.4f} std={stats['std']:.4f}"
        )

    if not args.no_plots:
        print(f"\nGenerating visualizations -> {args.output}/")
        base.generate_visualizations(
            table_ar,
            consistency_ar,
            completion_ar,
            clustering_ar,
            table_en,
            consistency_en,
            clustering_en,
            args.output,
        )

    output = {
        "experiment": "Measurement 4 — Pattern Geometry: Wazn as Geometric Operator",
        "framework": "AL-MIR'AH",
        "model": model_label,
        "english_model": en_model_label,
        "annotated_words": len(table_ar),
        "annotated_families": len(fpm_ar),
        "requested_families": len(families),
        "requested_annotations": len(annotations),
        "missing_arabic_embeddings": missing_ar,
        "annotation_source": annotation_meta.get("source", {}),
        "arabic_analogy_consistency": consistency_ar,
        "arabic_analogy_completion": [{k: v for k, v in r.items()} for r in completion_ar],
        "arabic_analogy_completion_summary": summarize_completion(completion_ar),
        "arabic_pattern_clustering": clustering_ar,
        "english_analogy_consistency": consistency_en,
        "english_pattern_clustering": clustering_en,
        "primary_triad_summary": {
            p: {
                "offset_consistency": consistency_ar.get(p, {}).get("mean_consistency"),
                "completion_mean": float(
                    np.mean([r["predicted_sim"] for r in completion_ar if r["pattern"] == p])
                )
                if any(r["pattern"] == p for r in completion_ar)
                else None,
                "pattern_gap": clustering_ar.get(p, {}).get("gap"),
            }
            for p in base.PRIMARY_TRIAD
        },
        "annotation_table": clean_table(table_ar),
    }

    path = Path(args.output) / "m4_results.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {path}")
    print("DONE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", default="m4_results_contextual_200")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-families", type=int, default=None)
    parser.add_argument("--skip-english", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
