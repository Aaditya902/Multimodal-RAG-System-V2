import hashlib
import json
import os
from typing import Optional
from config import config


class CaptionCacheService:
    def __init__(self, cache_path: str = ".caption_cache.json") -> None:
        self._path = cache_path
        self._cache: dict = self._load()

    def get(self, image_data: bytes) -> Optional[str]:
        """Return cached caption for this image, or None."""
        key = self._hash(image_data)
        return self._cache.get(key)

    def set(self, image_data: bytes, caption: str) -> None:
        """Store caption for this image and persist to disk."""
        key = self._hash(image_data)
        self._cache[key] = caption
        self._save()

    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        if os.path.exists(self._path):
            os.remove(self._path)

    @staticmethod
    def _hash(image_data: bytes) -> str:
        return hashlib.md5(image_data).hexdigest()

    def _load(self) -> dict:
        try:
            if os.path.exists(self._path):
                with open(self._path, "r") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    def _save(self) -> None:
        try:
            with open(self._path, "w") as f:
                json.dump(self._cache, f)
        except OSError:
            pass