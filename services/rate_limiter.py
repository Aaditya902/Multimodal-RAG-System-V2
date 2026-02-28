"""
RateLimiter: tracks API usage and enforces free tier limits.
Prevents hitting quota by refusing calls before they fail.
"""

import time
from collections import deque
from typing import Tuple
from config import config


class RateLimiter:
    """
    Dual-window rate limiter:
    - Per-minute: sliding window
    - Per-day: daily counter with midnight reset
    """

    def __init__(
        self,
        rpm_limit: int = config.rate_limit.max_requests_per_minute,
        daily_limit: int = config.rate_limit.max_requests_per_day,
    ) -> None:
        self._rpm_limit = rpm_limit
        self._daily_limit = daily_limit

        self._minute_window: deque = deque()  # timestamps of recent requests
        self._day_count: int = 0
        self._day_start: float = time.time()

    def can_proceed(self) -> Tuple[bool, str]:
        """
        Check if a request can proceed.
        Returns (allowed, reason_if_blocked).
        """
        now = time.time()
        self._reset_day_if_needed(now)
        self._clean_minute_window(now)

        if self._day_count >= self._daily_limit:
            remaining = 86400 - (now - self._day_start)
            hours = int(remaining // 3600)
            mins = int((remaining % 3600) // 60)
            return False, f"Daily limit reached. Resets in {hours}h {mins}m."

        if len(self._minute_window) >= self._rpm_limit:
            oldest = self._minute_window[0]
            wait = int(60 - (now - oldest)) + 1
            return False, f"Too many requests. Please wait {wait}s."

        return True, ""

    def record_request(self) -> None:
        """Call this after every successful API call."""
        now = time.time()
        self._minute_window.append(now)
        self._day_count += 1

    def stats(self) -> dict:
        """Return current usage stats for display."""
        now = time.time()
        self._reset_day_if_needed(now)
        self._clean_minute_window(now)
        day_remaining = 86400 - (now - self._day_start)

        return {
            "daily_used": self._day_count,
            "daily_limit": self._daily_limit,
            "daily_remaining": self._daily_limit - self._day_count,
            "rpm_used": len(self._minute_window),
            "rpm_limit": self._rpm_limit,
            "resets_in_hours": round(day_remaining / 3600, 1),
        }

    def _reset_day_if_needed(self, now: float) -> None:
        if now - self._day_start >= 86400:
            self._day_count = 0
            self._day_start = now

    def _clean_minute_window(self, now: float) -> None:
        while self._minute_window and now - self._minute_window[0] > 60:
            self._minute_window.popleft()