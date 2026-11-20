import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.chunking import Chunk

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # from the original RRF paper; results are not very sensitive to it


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _result(chunk: Chunk, score: float) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "source_file": chunk.source_file,
        "section": chunk.section,
        "score": float(score),
    }


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:top_k]
        return [_result(self.chunks[i], scores[i]) for i in order]


class DenseFAISSRetriever:
    # lightweight=True swaps the transformer encoder for TF-IDF vectors.
    # same FAISS index and interface, used in tests and CI where downloading
    # model weights would dominate the runtime.
    def __init__(self, chunks: list[Chunk], lightweight: bool = False, model_name: str = EMBED_MODEL):
        self.chunks = chunks
        self.lightweight = lightweight
        texts = [c.text for c in chunks]
        if lightweight:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(stop_words="english")
            matrix = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            matrix = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(matrix)
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        self._matrix = matrix

    def _query_vector(self, query: str) -> np.ndarray:
        if self.lightweight:
            vec = self.vectorizer.transform([query]).toarray().astype("float32")
        else:
            vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        scores, idx = self.index.search(self._query_vector(query), top_k)
        return [_result(self.chunks[i], scores[0][j]) for j, i in enumerate(idx[0]) if i >= 0]


def rrf_fuse(rankings: list[list[dict]], top_k: int = 5, k: int = RRF_K) -> list[dict]:
    fused: dict[str, dict] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            fused.setdefault(cid, item)
    order = sorted(scores, key=lambda cid: -scores[cid])[:top_k]
    return [{**fused[cid], "score": scores[cid]} for cid in order]


class HybridRRFRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseFAISSRetriever):
        self.bm25 = bm25
        self.dense = dense

    def search(self, query: str, top_k: int = 5, pool: int = 20) -> list[dict]:
        return rrf_fuse(
            [self.bm25.search(query, top_k=pool), self.dense.search(query, top_k=pool)],
            top_k=top_k,
        )


class CrossEncoderReranker:
    # lightweight=True scores by query-term coverage instead of the cross-encoder.
    # only used in tests; every reported number in the readme used the real model.
    def __init__(self, lightweight: bool = False, model_name: str = RERANK_MODEL):
        self.lightweight = lightweight
        self.model = None
        if not lightweight:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        if self.lightweight:
            q = set(tokenize(query))
            out = []
            for text in texts:
                words = set(tokenize(text))
                out.append(len(q & words) / max(len(q), 1))
            return out
        raw = self.model.predict([(query, t) for t in texts])
        # ms-marco cross-encoder outputs are logits; squash so thresholds mean the same thing in both modes
        return [float(1 / (1 + np.exp(-s))) for s in raw]

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return []
        scores = self.score_pairs(query, [c["text"] for c in candidates])
        ranked = [{**c, "score": s} for c, s in zip(candidates, scores)]
        return sorted(ranked, key=lambda r: -r["score"])[:top_k]
