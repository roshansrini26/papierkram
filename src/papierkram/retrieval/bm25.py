import re

from rank_bm25 import BM25Okapi

from papierkram.retrieval.models import Chunk

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase words. \\w matches umlauts and ß in Python, so 'Rundfunkbeitrag' stays one token."""
    return _WORD.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self._ids = [c.id for c in chunks]
        tokenized = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, n: int) -> list[str]:
        """Return the IDs of the n most relevant chunks for the query, in order of relevance."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._ids[i] for i in ranked[:n] if scores[i] > 0]
