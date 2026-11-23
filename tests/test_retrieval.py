from papierkram.retrieval.bm25 import BM25Index, tokenize
from papierkram.retrieval.fusion import interleave_by_source, reciprocal_rank_fusion
from papierkram.retrieval.hybrid import HybridRetriever
from tests.conftest import FakeEmbeddings, FakeVectorStore

# ---------- BM25 ----------


def test_tokenize_keeps_umlauts_and_lowercases():
    assert tokenize("Der BAföG-Antrag") == ["der", "bafög", "antrag"]


def test_bm25_finds_exact_term(sample_chunks):
    index = BM25Index(sample_chunks)
    assert index.search("Anmeldung Einzug", n=3)[0] == "anm-0"


def test_bm25_leaves_out_chunks_with_no_matching_word(sample_chunks):
    assert BM25Index(sample_chunks).search("Katze", n=5) == []


def test_bm25_on_empty_index_returns_nothing():
    assert BM25Index([]).search("anything", n=5) == []


# ---------- Reciprocal Rank Fusion ----------


def test_rrf_rewards_ids_found_by_both_retrievers():
    dense = ["a", "b", "c"]
    sparse = ["c", "d", "a"]
    merged = reciprocal_rank_fusion([dense, sparse])
    # "a" (ranks 1 and 3) and "c" (ranks 3 and 1) appear in both lists, so they beat "b" and "d".
    assert set(merged[:2]) == {"a", "c"}
    assert set(merged) == {"a", "b", "c", "d"}


def test_rrf_with_one_list_keeps_its_order():
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]


# ---------- Interleaving ----------


def test_interleave_stops_one_source_filling_every_slot(sample_chunks):
    # Ranked order puts all three rundfunkbeitrag chunks first.
    selected = interleave_by_source(sample_chunks, k=3)
    assert {c.source for c in selected} == {"rundfunkbeitrag", "tk", "anmeldung"}


def test_interleave_keeps_rank_order_within_a_source(sample_chunks):
    selected = interleave_by_source(sample_chunks, k=5)
    rb = [c.id for c in selected if c.source == "rundfunkbeitrag"]
    assert rb == ["rb-0", "rb-1", "rb-2"][: len(rb)]


def test_interleave_respects_allowed_sources(sample_chunks):
    selected = interleave_by_source(sample_chunks, k=5, allowed_sources={"tk"})
    assert [c.id for c in selected] == ["tk-0"]


def test_interleave_returns_fewer_than_k_when_chunks_run_out(sample_chunks):
    assert len(interleave_by_source(sample_chunks, k=50)) == len(sample_chunks)


# ---------- HybridRetriever end to end (with fakes) ----------


def test_hybrid_combines_dense_and_keyword_results(sample_chunks):
    # Dense search "finds" only the TK chunk; BM25 should add the Anmeldung chunk.
    store = FakeVectorStore(sample_chunks, dense_ranking=["tk-0"])
    retriever = HybridRetriever(store, FakeEmbeddings())

    results = retriever.retrieve("Anmeldung nach dem Einzug", k=2)
    assert {c.id for c in results} == {"tk-0", "anm-0"}


def test_hybrid_ignores_ids_missing_from_cache(sample_chunks):
    store = FakeVectorStore(sample_chunks, dense_ranking=["not-ingested-yet", "tk-0"])
    retriever = HybridRetriever(store, FakeEmbeddings())
    ids = [c.id for c in retriever.retrieve("Familienversicherung", k=3)]
    assert "not-ingested-yet" not in ids


def test_refresh_picks_up_new_chunks(sample_chunks):
    store = FakeVectorStore(sample_chunks[:1])
    retriever = HybridRetriever(store, FakeEmbeddings())
    assert retriever.size == 1

    store._chunks = sample_chunks
    retriever.refresh()
    assert retriever.size == len(sample_chunks)
