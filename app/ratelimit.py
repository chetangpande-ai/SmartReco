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


# Sized against real traffic, because the cost here is one token per *event*, not per
# request, and a single page view is worth about ten of them (product_view, four scroll
# marks, four impressions, a dwell). At capacity=60/refill=1 an engaged learner — the one
# whose profile is most worth building — was throttled after six pages and then held to
# one event a second. That is a data-loss bug wearing an abuse-control costume.
#
# 300 absorbs a full localStorage stash drain (MAX_STORED=200) in one go; 10/s sustained
# is still a hard ceiling a runaway loop hits immediately.
events_limiter = TokenBucket(capacity=300, refill_per_second=10.0)
auth_limiter = TokenBucket(capacity=10, refill_per_second=0.2)
recommend_limiter = TokenBucket(capacity=10, refill_per_second=0.1)
