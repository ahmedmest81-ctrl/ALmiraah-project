"""Regenerate Paper B raw-cosine demonstration baselines.

This script locks the four baseline cosine values used in Paper B §5.1-5.5.
It uses the same v3 embedding protocol as the coordinate engine:
CAMeLBERT-ca, three carrier sentences, layer-8 hidden states, target-word
span pooling, averaged across carriers. The baseline itself is ordinary
uncentered cosine between the two query embeddings.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
LAYER = 8
CARRIERS = ["الكلمة هي {w}", "يقول الشيخ {w}", "معنى {w} هو"]
DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")

PAIRS = [
    {
        "pair_id": "farah_surur",
        "term1_label": "faraḥ",
        "term1_ar": "فرح",
        "term2_label": "surūr",
        "term2_ar": "سرور",
        "manuscript_rounded": 0.937,
    },
    {
        "pair_id": "uhibbuka_ashaqaki",
        "term1_label": "uḥibbuka",
        "term1_ar": "أحبك",
        "term2_label": "aʿshaquki",
        "term2_ar": "أعشقك",
        "manuscript_rounded": 0.921,
    },
    {
        "pair_id": "samt_huzn",
        "term1_label": "ṣamt",
        "term1_ar": "صمت",
        "term2_label": "ḥuzn",
        "term2_ar": "حزن",
        "manuscript_rounded": 0.885,
    },
    {
        "pair_id": "ghadab_khawf",
        "term1_label": "ghaḍab",
        "term1_ar": "غضب",
        "term2_label": "khawf",
        "term2_ar": "خوف",
        "manuscript_rounded": 0.865,
    },
]


def strip_diacritics(text: str) -> str:
    return DIACRITICS.sub("", text)


def target_span(tokenizer, sentence: str, word: str) -> list[int]:
    start = sentence.index(word)
    end = start + len(word)
    enc = tokenizer(
        sentence,
        return_offsets_mapping=True,
        truncation=True,
        max_length=64,
    )
    span = [
        i
        for i, (a, b) in enumerate(enc["offset_mapping"])
        if a < end and b > start and b > a
    ]
    return span or list(range(1, len(enc["offset_mapping"]) - 1))


@torch.no_grad()
def embed_word(tokenizer, model, word: str, device: torch.device) -> np.ndarray:
    bare = strip_diacritics(word)
    vecs = []
    for carrier in CARRIERS:
        sentence = carrier.format(w=bare)
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = model(**inputs)
        hidden = output.hidden_states[LAYER][0]
        span = target_span(tokenizer, sentence, bare)
        vecs.append(hidden[span].mean(0).detach().cpu().numpy())
    return np.mean(vecs, axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper_b/baseline_cosines_v3.json"),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only the local Hugging Face cache; fail if the model is absent.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        output_hidden_states=True,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()

    terms = sorted({pair["term1_ar"] for pair in PAIRS} | {pair["term2_ar"] for pair in PAIRS})
    embeddings = {term: embed_word(tokenizer, model, term, device) for term in terms}

    pair_results = []
    for pair in PAIRS:
        actual = cosine(embeddings[pair["term1_ar"]], embeddings[pair["term2_ar"]])
        rounded = round(actual, 3)
        pair_results.append(
            {
                **pair,
                "raw_cosine": actual,
                "rounded_3dp": rounded,
                "matches_manuscript": rounded == pair["manuscript_rounded"],
            }
        )

    payload = {
        "artifact": "paper_b_raw_cosine_baselines",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "embedding_protocol": {
            "layer": LAYER,
            "carrier_sentences": CARRIERS,
            "pooling": "target-word subword span mean per carrier; mean across carriers",
            "cosine": "ordinary uncentered cosine between query embeddings",
            "diacritics": "Arabic short vowels/tatweel stripped before embedding",
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "pairs": pair_results,
        "all_match_manuscript": all(row["matches_manuscript"] for row in pair_results),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
