"""Retrieval: dense (Chroma), sparse (BM25) or hybrid (both, fused with Reciprocal Rank Fusion)."""

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.ingest import load_chunks, open_store

MODES = ("hybrid", "dense", "bm25")
CANDIDATES = 20  # per-ranker pool fed into RRF; the whole KB is ~30 chunks
BM25_WEIGHT = 0.5  # BM25 is a tie-breaker for exact terms (codes, acronyms), dense leads; see eval/results.md


# With ~30 short chunks, BM25 otherwise ranks a chunk highly just for repeating "the"/"is"/"how".
STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its my
of on or our the their there this to was what when where which who why will with you your""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens minus stopwords; numbers like "160,000" stay matchable as "160", "000"."""
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


def rrf(rankings: list[list[Document]], weights=None, c: int = 60) -> list[Document]:
    """Weighted Reciprocal Rank Fusion: score(d) = Σ w / (c + rank). Uses ranks only, so the
    incomparable scales of cosine similarity and BM25 never have to be normalised."""
    scores: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for ranking, w in zip(rankings, weights or [1.0] * len(rankings)):
        for rank, doc in enumerate(ranking, 1):
            scores[doc.id] = scores.get(doc.id, 0.0) + w / (c + rank)
            by_id[doc.id] = doc
    return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)]


class Retriever:
    """Holds both indexes over the same chunks and serves top-k results for a query."""

    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self.store = open_store(chunks)
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

    def dense(self, query: str, n: int) -> list[Document]:
        return self.store.similarity_search(query, k=n)

    def sparse(self, query: str, n: int) -> list[Document]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.chunks[i] for i in top if scores[i] > 0]  # no keyword overlap → not a hit

    def search(self, query: str, k: int, mode: str = "hybrid") -> list[Document]:
        if mode == "dense":
            return self.dense(query, k)
        if mode == "bm25":
            return self.sparse(query, k)
        n = max(k, CANDIDATES)  # each ranker contributes a wider candidate pool before fusion
        return rrf([self.dense(query, n), self.sparse(query, n)], [1.0, BM25_WEIGHT])[:k]


def load_retriever() -> Retriever:
    return Retriever(load_chunks())
