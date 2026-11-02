import time

from redis.asyncio import Redis


class RateLimiter:
    """Fixed-window per-user limiter backed by Redis (works across replicas)."""

    def __init__(self, redis: Redis, limit_per_minute: int) -> None:
        self._redis = redis
        self._limit = limit_per_minute

    async def hit(self, user_id: str) -> bool:
        """Record one request. Returns False when the user is over the limit."""
        key = f"rl:{user_id}:{int(time.time() // 60)}"
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 61)
        count, _ = await pipe.execute()
        return int(count) <= self._limit
