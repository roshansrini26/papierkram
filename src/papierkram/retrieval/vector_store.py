from pathlib import Path
from typing import Protocol

import chromadb

from papierkram.retrieval.models import Chunk


class VectorStore(Protocol):
    def all_chunks(self) -> list[Chunk]: ...

    def search(self, query_vector: list[float], n: int) -> list[str]: ...


class ChromaVectorStore:
    def __init__(self, path: Path, collection_name: str):
        client = chromadb.PersistentClient(path=str(path))
        self._collection = client.get_or_create_collection(name=collection_name)

    def all_chunks(self) -> list[Chunk]:
        items = self._collection.get(include=["documents", "metadatas"])
        return [
            Chunk(
                id=chunk_id,
                text=text,
                source=(meta or {}).get("source", "unknown"),
                chunk_index=(meta or {}).get("chunk_index", "unknown"),
            )
            for chunk_id, text, meta in zip(items["ids"], items["documents"], items["metadatas"], strict=True)
        ]

    def search(self, query_vector: list[float], n: int) -> list[str]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(query_embeddings=[query_vector], n_results=n)
        return result["ids"][0]
