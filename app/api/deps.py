from fastapi import Depends, HTTPException, status

from app.security.auth import require_user
from app.services import Services, get_services


async def rate_limited_user(
    user: str = Depends(require_user), services: Services = Depends(get_services)
) -> str:
    if services.rate_limiter and not await services.rate_limiter.hit(user):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
    return user
