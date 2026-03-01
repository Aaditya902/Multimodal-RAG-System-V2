import hashlib
import time
from typing import Optional
from core.models import QAResponse
from config import config



class CacheService:
    """Simple TTL-based in-memory cache for QA responses."""

    def __init__(self, ttl_seconds: int = config.rate_limit.cache_ttl_seconds) -> None:
        self._cache: dict = {}
        self._ttl = ttl_seconds

    def _key(self, query: str, model: str) -> str:
        """Generate a stable cache key from query + model."""
        raw = f"{query.strip().lower()}::{model}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, model: str) -> Optional[QAResponse]:
        """Return cached response if exists and not expired."""
        key = self._key(query, model)
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry["timestamp"] > self._ttl:
            del self._cache[key]
            return None
        return entry["response"]

    def set(self, query: str, model: str, response: QAResponse) -> None:
        """Cache a response."""
        key = self._key(query, model)
        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
        }

    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()