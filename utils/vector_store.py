import numpy as np
from typing import List, Dict, Tuple


class VectorStore:
    """In-memory FAISS-backed vector store with cosine similarity search."""

    def __init__(self):
        self._vectors: np.ndarray | None = None
        self._chunks: List[Dict] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        new_vecs = np.array(embeddings, dtype=np.float32)
        # L2-normalise for cosine similarity via dot product
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        new_vecs = new_vecs / norms

        if self._vectors is None:
            self._vectors = new_vecs
        else:
            self._vectors = np.vstack([self._vectors, new_vecs])

        self._chunks.extend(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        score_threshold: float = 0.25,
    ) -> List[Tuple[Dict, float]]:
        if self._vectors is None or len(self._chunks) == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        scores = self._vectors @ query  # cosine similarities
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= score_threshold:
                results.append((self._chunks[idx], score))

        return results

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._vectors = None
        self._chunks = []

    def remove_source(self, source: str) -> int:
        if not self._chunks:
            return 0
        keep_indices = [i for i, c in enumerate(self._chunks) if c["source"] != source]
        removed = len(self._chunks) - len(keep_indices)
        if keep_indices:
            self._vectors = self._vectors[keep_indices]
            self._chunks = [self._chunks[i] for i in keep_indices]
        else:
            self.clear()
        return removed

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def sources(self) -> List[str]:
        return sorted(set(c["source"] for c in self._chunks))
