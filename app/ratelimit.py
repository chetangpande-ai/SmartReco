"""Token bucket rate limiting, in-process.

A single-process deployment is the whole scope here, so a dict beats Redis. If this ever
runs multi-worker, swap the dict for Redis and the interface stays put.
"""

import threading
import time


class TokenBucket:
    def __init__(self, capacity: int, refill_per_second: float):
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_seen)
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_second)
            if tokens < cost:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - cost, now)
            return True

    def prune(self, older_than_seconds: float = 900.0) -> int:
        """Called by the scheduler so idle visitors don't leak memory forever."""
        cutoff = time.monotonic() - older_than_seconds
        with self._lock:
            stale = [k for k, (_, last) in self._buckets.items() if last < cutoff]
            for k in stale:
                del self._buckets[k]
            return len(stale)


# Generous: a real browsing session produces a few batched calls per minute, so this
# only bites on a runaway loop or an attempt to flood the events table.
events_limiter = TokenBucket(capacity=60, refill_per_second=1.0)
auth_limiter = TokenBucket(capacity=10, refill_per_second=0.2)
recommend_limiter = TokenBucket(capacity=10, refill_per_second=0.1)
