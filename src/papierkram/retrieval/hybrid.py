from langchain_core.embeddings import Embeddings

from papierkram.retrieval.bm25 import BM25Index
from papierkram.retrieval.fusion import interleave_by_source, reciprocal_rank_fusion
from papierkram.retrieval.models import Chunk
from papierkram.retrieval.vector_store import VectorStore


class HybridRetriever:
    def __init__(self, store: VectorStore, embeddings: Embeddings, candidates_per_retriever: int = 3):
        self._store = store
        self._embeddings = embeddings

        self._multiplier = candidates_per_retriever
        self.refresh()

    def refresh(self) -> None:
        chunks = self._store.all_chunks()
        self._chunks_by_id = {c.id: c for c in chunks}
        self._bm25 = BM25Index(chunks)

    @property
    def size(self) -> int:
        return len(self._chunks_by_id)

    def retrieve(self, query: str, k: int, allowed_sources: set[str] | None = None) -> list[Chunk]:
        n = k * self._multiplier
        dense_ids = self._store.search(self._embeddings.embed_query(query), n)
        sparse_ids = self._bm25.search(query, n)

        merged_ids = reciprocal_rank_fusion([dense_ids, sparse_ids])

        ranked = [self._chunks_by_id[i] for i in merged_ids if i in self._chunks_by_id]
        return interleave_by_source(ranked, k, allowed_sources)
