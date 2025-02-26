"""Tests for caching integration with JSONL backend."""

import json
from unittest.mock import call, patch

import pytest

from memory_mcp_server.backends.jsonl import JsonlBackend
from memory_mcp_server.interfaces import Entity, KnowledgeGraph, Relation


@pytest.fixture
def temp_jsonl_path(tmp_path):
    """Create a temporary file path for testing."""
    return tmp_path / "test_cache.jsonl"


@pytest.fixture
async def prepare_test_file(temp_jsonl_path):
    """Create a test file with some entities and relations."""
    with open(temp_jsonl_path, "w") as f:
        # Add some entities
        for i in range(5):
            entity = {
                "type": "entity",
                "name": f"entity{i}",
                "entityType": "person" if i % 2 == 0 else "project",
                "observations": [f"obs{i}_1", f"obs{i}_2"],
            }
            f.write(json.dumps(entity) + "\n")

        # Add some relations
        for i in range(3):
            relation = {
                "type": "relation",
                "from": f"entity{i}",
                "to": f"entity{i+1}",
                "relationType": "knows",
            }
            f.write(json.dumps(relation) + "\n")

    return temp_jsonl_path


@pytest.fixture
async def jsonl_backend(prepare_test_file):
    """Create and initialize a JsonlBackend instance."""
    backend = JsonlBackend(prepare_test_file)
    await backend.initialize()
    return backend


@pytest.mark.asyncio
async def test_graph_cache(jsonl_backend):
    """Test that graph is cached and used on subsequent reads."""

    # First read should load from file
    with patch.object(jsonl_backend, "_load_graph_with_memory_control") as mock_load:
        # Set up mock to return a real graph from file since we patch before the cache check
        # We need to use an async function that returns a proper graph
        async def mock_load_impl():
            return await jsonl_backend._load_graph_from_file()

        mock_load.side_effect = mock_load_impl

        # First read
        graph1 = await jsonl_backend.read_graph()

        # Verify loading was called
        mock_load.assert_called_once()

        # Reset mock for next call
        mock_load.reset_mock()

        # Second read should use cache
        graph2 = await jsonl_backend.read_graph()

        # Verify loading was not called again
        mock_load.assert_not_called()

        # Graphs should be the same object
        assert graph1 is graph2


@pytest.mark.asyncio
async def test_search_cache(jsonl_backend):
    """Test that search results are cached and used on subsequent searches."""

    # First patch the CacheManager's search_cache
    with patch.object(jsonl_backend.cache_manager.search_cache, "get") as mock_get:
        with patch.object(jsonl_backend.cache_manager.search_cache, "set") as mock_set:
            # Mock get to return None first (cache miss)
            mock_get.return_value = None

            # First search
            query = "test_query"
            await jsonl_backend.search_nodes(query)

            # Verify cache operations
            mock_get.assert_called_once()
            mock_set.assert_called_once()

            # Reset mocks
            mock_get.reset_mock()
            mock_set.reset_mock()

            # Mock get to return a cached result on second call
            mock_get.return_value = KnowledgeGraph(entities=[], relations=[])

            # Second search with same query
            await jsonl_backend.search_nodes(query)

            # Verify cache was checked but not set
            mock_get.assert_called_once()
            mock_set.assert_not_called()


@pytest.mark.asyncio
async def test_create_entities_invalidates_cache(jsonl_backend):
    """Test that creating entities invalidates relevant caches."""

    # Read initial graph to populate cache
    await jsonl_backend.read_graph()

    # Create test entities
    test_entities = [
        Entity(name="new_entity1", entityType="person", observations=["new_obs1"]),
        Entity(name="new_entity2", entityType="project", observations=["new_obs2"]),
    ]

    # Patch the cache invalidation methods
    with patch.object(
        jsonl_backend.cache_manager, "invalidate_entity"
    ) as mock_invalidate:
        with patch.object(
            jsonl_backend.cache_manager.graph_cache, "set"
        ) as mock_graph_set:
            # Create entities
            await jsonl_backend.create_entities(test_entities)

            # Verify cache operations
            assert mock_invalidate.call_count == 2
            mock_invalidate.assert_has_calls(
                [call("new_entity1"), call("new_entity2")], any_order=True
            )

            # Verify graph cache was updated
            mock_graph_set.assert_called_once()


@pytest.mark.asyncio
async def test_delete_entities_invalidates_cache(jsonl_backend):
    """Test that deleting entities invalidates relevant caches."""

    # First create entities to delete
    await jsonl_backend.create_entities(
        [Entity(name="delete_entity", entityType="person", observations=["delete_obs"])]
    )

    # Read graph to populate cache
    await jsonl_backend.read_graph()

    # Patch the cache invalidation methods
    with patch.object(
        jsonl_backend.cache_manager, "invalidate_entity"
    ) as mock_invalidate:
        with patch.object(
            jsonl_backend.cache_manager.graph_cache, "set"
        ) as mock_graph_set:
            # Delete entity
            await jsonl_backend.delete_entities(["delete_entity"])

            # Verify cache operations
            mock_invalidate.assert_called_once_with("delete_entity")
            mock_graph_set.assert_called_once()


@pytest.mark.asyncio
async def test_relations_invalidate_cache(jsonl_backend):
    """Test that creating and deleting relations invalidates relevant caches."""

    # Create test entities
    await jsonl_backend.create_entities(
        [
            Entity(name="rel_entity1", entityType="person", observations=[]),
            Entity(name="rel_entity2", entityType="project", observations=[]),
        ]
    )

    # Read graph to populate cache
    await jsonl_backend.read_graph()

    # Test creating relations - need to reset caching behavior before patching
    await jsonl_backend.read_graph()  # This ensures cache is populated

    # Need to check if the JSONL backend has caching update implemented
    # If not implemented, we'll skip the validation

    # Create relation without patching first
    test_relation = Relation(
        from_="rel_entity1", to="rel_entity2", relationType="knows"
    )
    await jsonl_backend.create_relations([test_relation])

    # Now test with patching for a second relation
    with patch.object(
        jsonl_backend.cache_manager, "invalidate_entity", return_value=None
    ) as mock_invalidate:
        with patch.object(
            jsonl_backend.cache_manager.graph_cache, "set"
        ) as mock_graph_set:
            # Create another relation
            await jsonl_backend.create_relations(
                [
                    Relation(
                        from_="rel_entity1", to="rel_entity2", relationType="created"
                    )
                ]
            )

            # The actual implementation might not call invalidate_entity in all cases
            # So we should only check if graph_cache.set was called
            mock_graph_set.assert_called_once()
            mock_invalidate.assert_has_calls(
                [call("rel_entity1"), call("rel_entity2")], any_order=True
            )
            mock_graph_set.assert_called_once()

            # Reset mocks
            mock_invalidate.reset_mock()
            mock_graph_set.reset_mock()

            # Delete relation
            await jsonl_backend.delete_relations("rel_entity1", "rel_entity2")

            # Verify cache operations
            assert mock_invalidate.call_count == 2
            mock_invalidate.assert_has_calls(
                [call("rel_entity1"), call("rel_entity2")], any_order=True
            )
            mock_graph_set.assert_called_once()


@pytest.mark.asyncio
async def test_add_observations_invalidates_cache(jsonl_backend):
    """Test that adding observations invalidates relevant caches."""

    # Create test entity
    await jsonl_backend.create_entities(
        [Entity(name="obs_entity", entityType="person", observations=["initial_obs"])]
    )

    # Read graph to populate cache
    await jsonl_backend.read_graph()

    # Patch the cache invalidation methods
    with patch.object(
        jsonl_backend.cache_manager, "invalidate_entity"
    ) as mock_invalidate:
        with patch.object(
            jsonl_backend.cache_manager.graph_cache, "set"
        ) as mock_graph_set:
            # Add observation
            await jsonl_backend.add_observations("obs_entity", ["new_obs"])

            # Verify cache operations
            mock_invalidate.assert_called_once_with("obs_entity")
            mock_graph_set.assert_called_once()


@pytest.mark.asyncio
async def test_flush_clears_all_caches(jsonl_backend):
    """Test that flush clears all caches."""

    # Read graph to populate cache
    await jsonl_backend.read_graph()

    # Make backend dirty to ensure flush does something
    jsonl_backend._dirty = True

    # Patch the cache clear method
    with patch.object(jsonl_backend.cache_manager, "clear_all") as mock_clear:
        # Flush
        await jsonl_backend.flush()

        # Verify cache operations
        mock_clear.assert_called_once()
