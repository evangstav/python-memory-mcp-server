import json
import pytest

from typing import List

# Adjust the imports according to your package structure
from memory_mcp_server.main import TOOLS, handle_error
from memory_mcp_server.interfaces import Entity, Relation, KnowledgeGraph
from memory_mcp_server.exceptions import EntityNotFoundError


@pytest.fixture
def mock_manager():
    class MockManager:
        async def create_entities(self, entities: List[Entity]):
            return entities

        async def create_relations(self, relations: List[Relation]):
            return relations

        async def add_observations(self, observations):
            return observations

        async def delete_entities(self, entity_names: List[str]):
            pass

        async def delete_observations(self, deletions):
            pass

        async def delete_relations(self, relations: List[Relation]):
            pass

        async def read_graph(self):
            # Return a simple graph
            return KnowledgeGraph(
                entities=[
                    Entity(
                        name="TestEntity", entityType="TypeA", observations=("obs1",)
                    )
                ],
                relations=[
                    Relation(
                        from_="TestEntity", to="AnotherEntity", relationType="knows"
                    )
                ],
            )

        async def search_nodes(self, query: str):
            # If the query matches "TestEntity", return the graph; otherwise empty
            if "TestEntity".lower() in query.lower():
                return await self.read_graph()
            return KnowledgeGraph(entities=[], relations=[])

        async def open_nodes(self, names: List[str]):
            # If "TestEntity" is requested, return it
            if "TestEntity" in names:
                return await self.read_graph()
            return KnowledgeGraph(entities=[], relations=[])

    return MockManager()


@pytest.mark.asyncio
async def test_create_entities(mock_manager):
    handler = TOOLS["create_entities"]
    arguments = {
        "entities": [
            {"name": "E1", "entityType": "TypeX", "observations": ["obsA"]},
            {"name": "E2", "entityType": "TypeY", "observations": ["obsB"]},
        ]
    }
    result = await handler(mock_manager, arguments)
    # result is a list of TextContent
    data = json.loads(result[0].text)
    assert len(data) == 2
    assert data[0]["name"] == "E1"
    assert data[1]["observations"] == ["obsB"]


@pytest.mark.asyncio
async def test_create_relations(mock_manager):
    handler = TOOLS["create_relations"]
    arguments = {"relations": [{"from": "E1", "to": "E2", "relationType": "likes"}]}
    result = await handler(mock_manager, arguments)
    data = json.loads(result[0].text)
    assert len(data) == 1
    assert data[0]["from"] == "E1"
    assert data[0]["to"] == "E2"


@pytest.mark.asyncio
async def test_add_observations(mock_manager):
    handler = TOOLS["add_observations"]
    arguments = {"observations": [{"entityName": "E1", "contents": ["newObs"]}]}
    result = await handler(mock_manager, arguments)
    data = json.loads(result[0].text)
    assert len(data) == 1
    assert data[0]["entityName"] == "E1"
    assert data[0]["contents"] == ["newObs"]


@pytest.mark.asyncio
async def test_delete_entities(mock_manager):
    handler = TOOLS["delete_entities"]
    arguments = {"entityNames": ["E1"]}
    result = await handler(mock_manager, arguments)
    assert "Entities deleted successfully" in result[0].text


@pytest.mark.asyncio
async def test_delete_observations(mock_manager):
    handler = TOOLS["delete_observations"]
    arguments = {"deletions": [{"entityName": "E1", "observations": ["obs1"]}]}
    result = await handler(mock_manager, arguments)
    assert "Observations deleted successfully" in result[0].text


@pytest.mark.asyncio
async def test_delete_relations(mock_manager):
    handler = TOOLS["delete_relations"]
    arguments = {"relations": [{"from": "E1", "to": "E2", "relationType": "knows"}]}
    result = await handler(mock_manager, arguments)
    assert "Relations deleted successfully" in result[0].text


@pytest.mark.asyncio
async def test_read_graph(mock_manager):
    handler = TOOLS["read_graph"]
    arguments = {}
    result = await handler(mock_manager, arguments)
    graph = json.loads(result[0].text)
    assert len(graph["entities"]) == 1
    assert graph["entities"][0]["name"] == "TestEntity"


@pytest.mark.asyncio
async def test_search_nodes(mock_manager):
    handler = TOOLS["search_nodes"]
    arguments = {"query": "TestEntity"}
    result = await handler(mock_manager, arguments)
    graph = json.loads(result[0].text)
    assert len(graph["entities"]) == 1
    assert graph["entities"][0]["name"] == "TestEntity"


@pytest.mark.asyncio
async def test_open_nodes(mock_manager):
    handler = TOOLS["open_nodes"]
    arguments = {"names": ["TestEntity"]}
    result = await handler(mock_manager, arguments)
    graph = json.loads(result[0].text)
    assert len(graph["entities"]) == 1
    assert graph["entities"][0]["name"] == "TestEntity"


def test_error_handling():
    msg = handle_error(EntityNotFoundError("MissingEntity"))
    assert "Entity not found: MissingEntity" in msg
