from app.rag.models import RetrievedChunk
from app.rag.retriever import reciprocal_rank_fusion
from app.rag.store import to_or_tsquery


def _c(i: int) -> RetrievedChunk:
    return RetrievedChunk(id=i, doc_name="d", content=f"c{i}")


def test_rrf_rewards_agreement_between_retrievers():
    dense = [_c(1), _c(2), _c(3)]
    sparse = [_c(3), _c(4), _c(1)]
    fused = [c.id for c in reciprocal_rank_fusion([dense, sparse])]
    assert fused[:2] == [1, 3]  # appear in both lists
    assert set(fused) == {1, 2, 3, 4}


def test_or_tsquery_sanitizes_input():
    q = to_or_tsquery("Phí thẻ Gold?! ' OR 1=1 --")
    assert "'" not in q and "-" not in q
    assert set(q.split(" | ")) >= {"phí", "thẻ", "gold", "or"}


async def test_hybrid_search_finds_relevant_chunk(retriever):
    results = await retriever.search("phí thường niên thẻ Gold")
    assert results[0].content.startswith("Phí thường niên thẻ tín dụng Gold")
