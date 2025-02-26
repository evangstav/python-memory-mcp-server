"""Caching system for Memory MCP Server."""

import time
from typing import Any, Dict, Generic, Optional, TypeVar

from memory_mcp_server.interfaces import Entity, KnowledgeGraph

K = TypeVar("K")
V = TypeVar("V")


class CacheEntry(Generic[V]):
    """A single cache entry with expiration time."""

    def __init__(self, value: V, ttl_seconds: float):
        self.value = value
        self.expiration = time.monotonic() + ttl_seconds

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.monotonic() > self.expiration


class Cache(Generic[K, V]):
    """Generic cache with TTL."""

    def __init__(
        self, default_ttl_seconds: float = 60.0, max_size: Optional[int] = None
    ):
        self.default_ttl = default_ttl_seconds
        self.max_size = max_size
        self.cache: Dict[K, CacheEntry[V]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> Optional[V]:
        """Get value from cache, None if not found or expired."""
        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]
        if entry.is_expired():
            del self.cache[key]
            self.misses += 1
            return None

        self.hits += 1
        return entry.value

    def set(self, key: K, value: V, ttl_seconds: Optional[float] = None) -> None:
        """Set value in cache with optional TTL."""
        # If cache is full, evict oldest entry
        if self.max_size and len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].expiration)
            del self.cache[oldest_key]

        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        self.cache[key] = CacheEntry(value, ttl)

    def delete(self, key: K) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        now = time.monotonic()
        expired_keys = [k for k, v in self.cache.items() if v.expiration < now]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0


class CacheManager:
    """Manages multiple caches for different operations."""

    def __init__(self):
        # Graph level cache
        self.graph_cache = Cache[str, KnowledgeGraph](default_ttl_seconds=60.0)

        # Entity level cache
        self.entity_cache = Cache[str, Entity](default_ttl_seconds=60.0, max_size=1000)

        # Search query cache
        self.search_cache = Cache[str, KnowledgeGraph](
            default_ttl_seconds=30.0, max_size=100
        )

    def clear_all(self) -> None:
        """Clear all caches."""
        self.graph_cache.clear()
        self.entity_cache.clear()
        self.search_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "graph_cache": {
                "size": self.graph_cache.size,
                "hit_rate": self.graph_cache.hit_rate,
                "hits": self.graph_cache.hits,
                "misses": self.graph_cache.misses,
            },
            "entity_cache": {
                "size": self.entity_cache.size,
                "hit_rate": self.entity_cache.hit_rate,
                "hits": self.entity_cache.hits,
                "misses": self.entity_cache.misses,
            },
            "search_cache": {
                "size": self.search_cache.size,
                "hit_rate": self.search_cache.hit_rate,
                "hits": self.search_cache.hits,
                "misses": self.search_cache.misses,
            },
        }

    def invalidate_entity(self, entity_name: str) -> None:
        """Invalidate cache entries related to an entity."""
        self.entity_cache.delete(entity_name)
        self.graph_cache.clear()  # Simply clear the graph cache
        # Search cache is more complex - for now just clear it entirely
        self.search_cache.clear()
