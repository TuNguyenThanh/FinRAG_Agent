from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings


async def require_user(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> str:
    """Resolve the caller from `X-API-Key`. Returns the user id."""
    user = settings.api_key_map().get(x_api_key or "")
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")
    return user
