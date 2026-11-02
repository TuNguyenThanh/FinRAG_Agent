import fakeredis

from app.core.cache import SemanticCache
from app.rag.embeddings import HashEmbedder


async def test_semantic_cache_hit_and_miss():
    cache = SemanticCache(
        fakeredis.FakeAsyncRedis(decode_responses=True), HashEmbedder(), 0.9, 60, 10
    )
    await cache.store("phí thường niên thẻ gold", "500.000đ", [{"doc_name": "a"}])
    hit = await cache.lookup("phí thường niên thẻ gold")
    assert hit and hit.answer == "500.000đ"
    assert await cache.lookup("lãi suất vay mua nhà") is None


async def test_cache_respects_max_entries_and_clear():
    cache = SemanticCache(
        fakeredis.FakeAsyncRedis(decode_responses=True), HashEmbedder(), 0.9, 60, 2
    )
    for i in range(5):
        await cache.store(f"câu hỏi số {i}", str(i), [])
    assert await cache.lookup("câu hỏi số 0") is None  # evicted
    assert (await cache.lookup("câu hỏi số 4")).answer == "4"
    await cache.clear()
    assert await cache.lookup("câu hỏi số 4") is None
