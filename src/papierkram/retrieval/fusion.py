from collections import defaultdict

from papierkram.retrieval.models import Chunk


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Merge several ranked id lists into one.

    Each id gets 1 / (k + rank) from every list it appears in. RRF uses ranks,
    not scores, because BM25 scores and cosine distances are on different scales.
    k = 60 is the value from the original RRF paper; it stops the top rank from dominating.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1 / (k + rank)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


def interleave_by_source(chunks: list[Chunk], k: int, allowed_sources: set[str] | None = None) -> list[Chunk]:
    """Pick up to k chunks, taking one from each source in turn (round robin),
    so a single long document cannot fill every slot.

    `chunks` must already be in ranked order; that order is kept within each source.
    """
    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if allowed_sources and chunk.source not in allowed_sources:
            continue
        by_source[chunk.source].append(chunk)

    selected: list[Chunk] = []
    round_idx = 0
    while len(selected) < k:
        added = False
        for source_chunks in by_source.values():
            if round_idx < len(source_chunks):
                selected.append(source_chunks[round_idx])
                added = True
                if len(selected) == k:
                    break
        if not added:  # every source is used up
            break
        round_idx += 1
    return selected
