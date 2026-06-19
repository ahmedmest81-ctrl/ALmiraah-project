"""
Small AL-MIR'AH RAG helper.

This is intentionally lightweight: it retrieves supporting passages from a
local Arabic corpus and adds AL-MIR'AH-style diagnostics instead of calling an
LLM. Use it as the evidence layer that a future answer generator can cite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
except Exception:  # pragma: no cover - fallback is for minimal environments
    TfidfVectorizer = None
    linear_kernel = None


LOG = logging.getLogger("almiraah_rag")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MIN_SENTENCE_TOKENS = 6
MAX_SENTENCE_TOKENS = 80

ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
NON_ARABIC_RE = re.compile(r"[^\u0600-\u06FF\s]")
SENTENCE_SPLIT_RE = re.compile(r"[.\n\u060C\u061B\u061F]+")

ABJAD_VALUES = {
    "\u0627": 1,
    "\u0628": 2,
    "\u062C": 3,
    "\u062F": 4,
    "\u0647": 5,
    "\u0648": 6,
    "\u0632": 7,
    "\u062D": 8,
    "\u0637": 9,
    "\u064A": 10,
    "\u0643": 20,
    "\u0644": 30,
    "\u0645": 40,
    "\u0646": 50,
    "\u0633": 60,
    "\u0639": 70,
    "\u0641": 80,
    "\u0635": 90,
    "\u0642": 100,
    "\u0631": 200,
    "\u0634": 300,
    "\u062A": 400,
    "\u062B": 500,
    "\u062E": 600,
    "\u0630": 700,
    "\u0636": 800,
    "\u0638": 900,
    "\u063A": 1000,
    "\u0623": 1,
    "\u0625": 1,
    "\u0622": 1,
    "\u0629": 400,
    "\u0649": 10,
    "\u0624": 6,
    "\u0626": 10,
}

ARABIC_PREFIXES = [
    "\u0648\u0627\u0644",
    "\u0641\u0627\u0644",
    "\u0628\u0627\u0644",
    "\u0644\u0644",
    "\u0627\u0644",
    "\u0648",
    "\u0641",
    "\u0628",
    "\u0644",
    "\u0643",
]

WEAK_LETTERS = {
    "\u0627",
    "\u0648",
    "\u064A",
    "\u0649",
    "\u0629",
    "\u0621",
    "\u0623",
    "\u0625",
    "\u0622",
    "\u0624",
    "\u0626",
}

CURATED_ROOTS = {
    "\u0627\u0644\u0644\u0647": "\u0627\u0644\u0647",
    "\u0648\u062C\u0648\u062F": "\u0648\u062C\u062F",
    "\u0639\u0627\u0644\u0645": "\u0639\u0644\u0645",
    "\u0639\u0644\u0645": "\u0639\u0644\u0645",
    "\u062D\u0642": "\u062D\u0642\u0642",
    "\u0648\u0627\u062D\u062F": "\u0648\u062D\u062F",
    "\u0630\u0627\u062A": "\u0630\u0627\u062A",
    "\u0646\u0648\u0631": "\u0646\u0648\u0631",
    "\u0631\u0648\u062D": "\u0631\u0648\u062D",
    "\u0642\u0644\u0628": "\u0642\u0644\u0628",
    "\u0639\u0642\u0644": "\u0639\u0642\u0644",
    "\u0646\u0641\u0633": "\u0646\u0641\u0633",
    "\u062D\u0643\u0645\u0629": "\u062D\u0643\u0645",
    "\u0645\u0639\u0631\u0641\u0629": "\u0639\u0631\u0641",
    "\u0645\u062D\u0628\u0629": "\u062D\u0628\u0628",
    "\u0641\u0646\u0627\u0621": "\u0641\u0646\u064A",
    "\u0628\u0642\u0627\u0621": "\u0628\u0642\u064A",
    "\u062D\u0636\u0631\u0629": "\u062D\u0636\u0631",
    "\u0645\u0642\u0627\u0645": "\u0642\u0648\u0645",
    "\u0634\u0647\u0648\u062F": "\u0634\u0647\u062F",
    "\u0643\u0634\u0641": "\u0643\u0634\u0641",
    "\u062E\u0644\u0642": "\u062E\u0644\u0642",
    "\u0631\u062D\u0645\u0629": "\u0631\u062D\u0645",
    "\u0625\u064A\u0645\u0627\u0646": "\u0627\u0645\u0646",
    "\u0625\u0633\u0644\u0627\u0645": "\u0633\u0644\u0645",
    "\u0639\u0628\u0627\u062F\u0629": "\u0639\u0628\u062F",
    "\u0635\u0628\u0631": "\u0635\u0628\u0631",
    "\u0634\u0643\u0631": "\u0634\u0643\u0631",
}


@dataclass(frozen=True)
class Passage:
    id: str
    index: int
    start_sentence: int
    end_sentence: int
    text: str


def clean_arabic(text: str) -> str:
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = NON_ARABIC_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    cleaned = clean_arabic(text)
    if not cleaned:
        return []
    return cleaned.split()


def abjad_value(word: str) -> int:
    return sum(ABJAD_VALUES.get(ch, 0) for ch in clean_arabic(word))


def extract_root_heuristic(word: str) -> str:
    word = clean_arabic(word)
    if not word:
        return ""
    if word in CURATED_ROOTS:
        return CURATED_ROOTS[word]

    for prefix in ARABIC_PREFIXES:
        if word.startswith(prefix) and len(word) > len(prefix) + 2:
            word = word[len(prefix) :]
            break

    consonants = [ch for ch in word if ch not in WEAK_LETTERS and ch in ABJAD_VALUES]
    if len(consonants) < 3:
        return ""
    return "".join(consonants[:3])


def load_sentences(path: Path, min_tokens: int, max_tokens: int) -> list[str]:
    raw_text = path.read_text(encoding="utf-8")
    sentences = []
    for raw_sentence in SENTENCE_SPLIT_RE.split(raw_text):
        sentence = clean_arabic(raw_sentence)
        n_tokens = len(sentence.split())
        if min_tokens <= n_tokens <= max_tokens:
            sentences.append(sentence)
    return sentences


def passage_id(text: str, index: int) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"p{index:06d}_{digest}"


def build_passages(
    sentences: list[str],
    window: int,
    max_passages: int | None,
) -> list[Passage]:
    if not sentences:
        return []

    limit = len(sentences) if max_passages is None else min(max_passages, len(sentences))
    passages = []
    for idx in range(limit):
        start = max(0, idx - window)
        end = min(len(sentences), idx + window + 1)
        text = " ".join(sentences[start:end])
        passages.append(
            Passage(
                id=passage_id(text, idx),
                index=idx,
                start_sentence=start,
                end_sentence=end - 1,
                text=text,
            )
        )
    return passages


def lexical_score(query_tokens: set[str], passage_tokens: set[str]) -> float:
    if not query_tokens or not passage_tokens:
        return 0.0
    overlap = query_tokens & passage_tokens
    return len(overlap) / max(1, len(query_tokens))


def retrieve_bm25(query: str, passages: list[Passage], top_k: int) -> list[tuple[int, float]]:
    tokenized_docs = [tokenize(p.text) for p in passages]
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    doc_freq = Counter()
    for doc_tokens in tokenized_docs:
        doc_freq.update(set(doc_tokens))

    n_docs = len(tokenized_docs)
    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / max(1, n_docs)
    k1 = 1.5
    b = 0.75
    scored = []

    for idx, doc_tokens in enumerate(tokenized_docs):
        counts = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        score = 0.0
        for term in query_tokens:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * doc_len / max(1.0, avg_doc_len))
            score += idf * (tf * (k1 + 1.0)) / max(denom, 1e-9)
        scored.append((idx, score))

    return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def retrieve_tfidf(query: str, passages: list[Passage], top_k: int) -> list[tuple[int, float]]:
    if TfidfVectorizer is None or linear_kernel is None:
        LOG.warning("scikit-learn is unavailable; falling back to built-in BM25 retrieval.")
        return retrieve_bm25(query, passages, top_k)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=1,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([p.text for p in passages])
    query_vector = vectorizer.transform([clean_arabic(query)])
    scores = linear_kernel(query_vector, matrix).ravel()
    top_indices = scores.argsort()[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def nearest_abjad_links(query_tokens: Iterable[str], passage_tokens: Iterable[str]) -> list[dict]:
    passage_values = [(token, abjad_value(token)) for token in passage_tokens]
    links = []
    for q_token in query_tokens:
        q_value = abjad_value(q_token)
        if q_value == 0 or not passage_values:
            continue
        nearest_token, nearest_value = min(
            passage_values,
            key=lambda item: abs(item[1] - q_value),
        )
        links.append(
            {
                "query_token": q_token,
                "query_abjad": q_value,
                "nearest_passage_token": nearest_token,
                "nearest_passage_abjad": nearest_value,
                "delta": abs(nearest_value - q_value),
            }
        )
    return sorted(links, key=lambda item: item["delta"])[:8]


def explain_passage(query: str, passage: Passage, score: float) -> dict:
    query_tokens = tokenize(query)
    passage_tokens = tokenize(passage.text)
    query_set = set(query_tokens)
    passage_set = set(passage_tokens)

    query_roots = {token: extract_root_heuristic(token) for token in query_tokens}
    passage_root_hits = []
    query_root_values = {root for root in query_roots.values() if root}
    for token in passage_set:
        root = extract_root_heuristic(token)
        if root and root in query_root_values:
            passage_root_hits.append({"token": token, "root": root})

    return {
        "id": passage.id,
        "score": round(score, 6),
        "sentence_span": [passage.start_sentence, passage.end_sentence],
        "token_overlap": sorted(query_set & passage_set),
        "query_profile": [
            {
                "token": token,
                "root": query_roots[token],
                "abjad": abjad_value(token),
            }
            for token in query_tokens
        ],
        "shared_root_hits": sorted(passage_root_hits, key=lambda item: item["token"])[:12],
        "nearest_abjad_links": nearest_abjad_links(query_tokens, passage_tokens),
        "text": passage.text,
    }


def write_report(output_path: Path, query: str, results: list[dict]) -> None:
    lines = [
        "# AL-MIR'AH RAG Retrieval Report",
        "",
        f"Query: {query}",
        "",
    ]
    for rank, item in enumerate(results, start=1):
        lines.extend(
            [
                f"## {rank}. {item['id']} | score={item['score']}",
                "",
                f"Sentence span: {item['sentence_span'][0]}-{item['sentence_span'][1]}",
                "",
                "Token overlap: " + (", ".join(item["token_overlap"]) or "none"),
                "",
                "Shared root hits: "
                + (
                    ", ".join(f"{hit['token']}:{hit['root']}" for hit in item["shared_root_hits"])
                    or "none"
                ),
                "",
                "Nearest Abjad links:",
            ]
        )
        for link in item["nearest_abjad_links"]:
            lines.append(
                f"- {link['query_token']}({link['query_abjad']}) -> "
                f"{link['nearest_passage_token']}({link['nearest_passage_abjad']}), "
                f"delta={link['delta']}"
            )
        lines.extend(["", item["text"], ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_rag(args: argparse.Namespace) -> dict:
    sentences = load_sentences(Path(args.corpus), args.min_tokens, args.max_tokens)
    if args.n_sentences:
        sentences = sentences[: args.n_sentences]
    passages = build_passages(sentences, args.window, args.max_passages)
    if not passages:
        raise ValueError("No passages were built from the corpus.")

    LOG.info("Loaded %s sentences and built %s passages", len(sentences), len(passages))

    retrieved = retrieve_tfidf(args.query, passages, args.top_k)
    results = [
        explain_passage(args.query, passages[index], score)
        for index, score in retrieved
        if score > 0 or args.include_zero_scores
    ]

    payload = {
        "query": args.query,
        "cleaned_query": clean_arabic(args.query),
        "corpus": str(Path(args.corpus).resolve()),
        "settings": {
            "top_k": args.top_k,
            "window": args.window,
            "n_sentences": args.n_sentences,
            "max_passages": args.max_passages,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
        },
        "results": results,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small AL-MIR'AH RAG retriever with Abjad/root diagnostics."
    )
    parser.add_argument("--corpus", default="corpus.txt", help="Arabic corpus text file.")
    parser.add_argument("--query", required=True, help="Arabic query or concept to retrieve around.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of passages to return.")
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Neighboring sentences to include on each side of a hit sentence.",
    )
    parser.add_argument(
        "--n-sentences",
        type=int,
        default=20000,
        help="Sentence limit before passage building. Use 0 for all sentences.",
    )
    parser.add_argument(
        "--max-passages",
        type=int,
        default=12000,
        help="Maximum number of passage windows to index. Use 0 for all passages.",
    )
    parser.add_argument("--min-tokens", type=int, default=MIN_SENTENCE_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=MAX_SENTENCE_TOKENS)
    parser.add_argument(
        "--include-zero-scores",
        action="store_true",
        help="Keep empty/zero-score results in the JSON output.",
    )
    parser.add_argument("--output-json", default="rag_results/almiraah_rag_results.json")
    parser.add_argument("--output-report", default="rag_results/almiraah_rag_report.md")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    if args.n_sentences == 0:
        args.n_sentences = None
    if args.max_passages == 0:
        args.max_passages = None

    payload = run_rag(args)

    output_json = Path(args.output_json)
    output_report = Path(args.output_report)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_report, payload["query"], payload["results"])

    print(f"Wrote JSON: {output_json}")
    print(f"Wrote report: {output_report}")
    for rank, result in enumerate(payload["results"], start=1):
        preview = result["text"][:180].replace("\n", " ")
        print(f"{rank}. score={result['score']} {result['id']} :: {preview}")


if __name__ == "__main__":
    main()
