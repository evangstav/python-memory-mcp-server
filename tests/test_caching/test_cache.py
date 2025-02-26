"""Tests for cache implementation."""

import time

from memory_mcp_server.caching.cache import Cache, CacheEntry, CacheManager
from memory_mcp_server.interfaces import Entity, KnowledgeGraph


def test_cache_entry():
    # Test initialization
    entry = CacheEntry("test_value", 0.1)
    assert entry.value == "test_value"

    # Test expiration
    assert not entry.is_expired()  # Should not be expired immediately
    time.sleep(0.2)
    assert entry.is_expired()  # Should be expired after waiting


def test_cache_basic_operations():
    cache = Cache[str, str](default_ttl_seconds=0.2)

    # Test set and get
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"

    # Test get with non-existent key
    assert cache.get("non_existent") is None

    # Test TTL expiration
    time.sleep(0.3)
    assert cache.get("key1") is None

    # Test custom TTL
    cache.set("key2", "value2", ttl_seconds=1.0)
    assert cache.get("key2") == "value2"

    # Test delete
    assert cache.delete("key2")
    assert cache.get("key2") is None

    # Test delete non-existent key
    assert not cache.delete("non_existent")


def test_cache_max_size():
    # Create cache with max size of 2
    cache = Cache[str, str](default_ttl_seconds=10.0, max_size=2)

    # Add 3 entries (should evict the oldest)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")

    # First key should be evicted
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


def test_cache_hit_rate():
    cache = Cache[str, str]()

    # Initial hit rate should be 0
    assert cache.hit_rate == 0

    # Add an entry
    cache.set("key", "value")

    # Hit
    assert cache.get("key") == "value"

    # Miss
    assert cache.get("missing") is None

    # Hit rate should be 50%
    assert cache.hit_rate == 50.0

    # Another hit
    assert cache.get("key") == "value"

    # Hit rate should be 66.67%
    assert round(cache.hit_rate, 2) == 66.67


def test_cache_cleanup_expired():
    cache = Cache[str, str](default_ttl_seconds=0.1)

    # Add entries
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")

    # Initial size should be 3
    assert cache.size == 3

    time.sleep(0.2)

    # Cleanup expired entries
    removed = cache.cleanup_expired()
    assert removed == 3
    assert cache.size == 0


def test_cache_manager():
    manager = CacheManager()

    # Test graph cache
    graph = KnowledgeGraph(entities=[], relations=[])
    manager.graph_cache.set("test_graph", graph)
    assert manager.graph_cache.get("test_graph") == graph

    # Test entity cache
    entity = Entity(name="test", entityType="person", observations=[])
    manager.entity_cache.set("test_entity", entity)
    assert manager.entity_cache.get("test_entity") == entity

    # Test search cache
    search_result = KnowledgeGraph(entities=[entity], relations=[])
    manager.search_cache.set("test_query", search_result)
    assert manager.search_cache.get("test_query") == search_result


def test_cache_manager_invalidation():
    manager = CacheManager()

    # Set up cache entries
    entity = Entity(name="test", entityType="person", observations=[])
    search_result = KnowledgeGraph(entities=[entity], relations=[])
    graph = KnowledgeGraph(entities=[entity], relations=[])

    manager.entity_cache.set("test", entity)
    manager.search_cache.set("query", search_result)
    manager.graph_cache.set("full_graph", graph)

    # Invalidate entity
    manager.invalidate_entity("test")

    # Entity cache should be empty for that key
    assert manager.entity_cache.get("test") is None

    # Search cache should be cleared entirely
    assert manager.search_cache.size == 0

    # Graph cache should be cleared entirely
    assert manager.graph_cache.size == 0


def test_cache_manager_clear_all():
    manager = CacheManager()

    # Set up cache entries
    manager.entity_cache.set("key1", "value1")
    manager.search_cache.set("key2", "value2")
    manager.graph_cache.set("key3", "value3")

    # Clear all caches
    manager.clear_all()

    # All caches should be empty
    assert manager.entity_cache.size == 0
    assert manager.search_cache.size == 0
    assert manager.graph_cache.size == 0


def test_cache_manager_stats():
    manager = CacheManager()

    # Set up cache entries and simulate hits/misses
    manager.entity_cache.set("key1", "value1")
    manager.entity_cache.get("key1")  # Hit
    manager.entity_cache.get("missing")  # Miss

    # Get stats
    stats = manager.get_stats()

    # Check that all caches have stats
    assert "graph_cache" in stats
    assert "entity_cache" in stats
    assert "search_cache" in stats

    # Check entity cache stats
    assert stats["entity_cache"]["size"] == 1
    assert stats["entity_cache"]["hits"] == 1
    assert stats["entity_cache"]["misses"] == 1
    assert stats["entity_cache"]["hit_rate"] == 50.0
