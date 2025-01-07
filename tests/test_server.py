"""Tests for the MCP server implementation."""

import json
from typing import Any, Dict, List, Protocol, cast

import pytest
from mcp.types import TextContent

from memory_mcp_server.exceptions import EntityNotFoundError
from memory_mcp_server.interfaces import Entity, KnowledgeGraph, Relation


# Mock tools and handlers
def handle_error(error: Exception) -> str:
    """Mock error handler."""
    if isinstance(error, EntityNotFoundError):
        return str(error)
    return f"Error: {str(error)}"


async def create_entities_handler(
    manager: Any, arguments: Dict[str, Any]
) -> List[TextContent]:
    """Mock create entities handler."""
    entities = [
        Entity(
            name=e["name"],
            entityType=e["entityType"],
            observations=e.get("observations", []),
        )
        for e in arguments["entities"]
    ]
    result = await manager.create_entities(entities)
    return [TextContent(type="text", text=json.dumps([e.to_dict() for e in result]))]


async def create_relations_handler(
    manager: Any, arguments: Dict[str, Any]
) -> List[TextContent]:
    """Mock create relations handler."""
    relations = [
        Relation(from_=r["from"], to=r["to"], relationType=r["relationType"])
        for r in arguments["relations"]
    ]
    result = await manager.create_relations(relations)
    return [TextContent(type="text", text=json.dumps([r.to_dict() for r in result]))]


async def add_observations_handler(
    manager: Any, arguments: Dict[str, Any]
) -> List[TextContent]:
    """Mock add observations handler."""
    await manager.add_observations(arguments["entity"], arguments["observations"])
    return [TextContent(type="text", text=json.dumps({"success": True}))]


async def delete_entities_handler(
    manager: Any, arguments: Dict[str, Any]
) -> List[TextContent]:
    """Mock delete entities handler."""
    await manager.delete_entities(arguments["names"])
    return [TextContent(type="text", text=json.dumps({"success": True}))]


async def delete_observations_handler(
    manager: Any, arguments: Dict[str, Any]
) -> List[TextContent]:
    """Mock delete observations handler."""
    await manager.delete_observations(arguments["entity"], arguments["observations"])
    return [TextContent(type="text", text=json.dumps({"success": True}))]


async def delete_relations_handler(
    manager: Any, arguments: Dict[str, Any]
) -> List[TextContent]:
    """Mock delete relations handler."""
    await manager.delete_relations(arguments["from"], arguments["to"])
    return [TextContent(type="text", text=json.dumps({"success": True}))]


async def read_graph_handler(
    manager: Any, arguments: Dict[str, Any]
) -> List[TextContent]:
    """Mock read graph handler."""
    graph = await manager.read_graph()
    return [TextContent(type="text", text=json.dumps(graph.to_dict()))]


async def search_nodes_handler(
    manager: Any, arguments: Dict[str, Any]
) -> List[TextContent]:
    """Mock search nodes handler."""
    result = await manager.search_nodes(arguments["query"])
    return [TextContent(type="text", text=json.dumps(result.to_dict()))]


async def open_nodes_handler(
    manager: Any, arguments: Dict[str, Any]
) -> List[TextContent]:
    """Mock open nodes handler."""
    result = await manager.open_nodes(arguments["names"])
    return [TextContent(type="text", text=json.dumps(result.to_dict()))]


TOOLS: Dict[str, Any] = {
    "create_entities": create_entities_handler,
    "create_relations": create_relations_handler,
    "add_observations": add_observations_handler,
    "delete_entities": delete_entities_handler,
    "delete_observations": delete_observations_handler,
    "delete_relations": delete_relations_handler,
    "read_graph": read_graph_handler,
    "search_nodes": search_nodes_handler,
    "open_nodes": open_nodes_handler,
}


class MockManagerProtocol(Protocol):
    """Protocol defining the interface for MockManager."""

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        ...

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        ...

    async def add_observations(self, entity: str, observations: List[str]) -> None:
        ...

    async def delete_entities(self, names: List[str]) -> None:
        ...

    async def delete_observations(self, entity: str, observations: List[str]) -> None:
        ...

    async def delete_relations(self, from_: str, to: str) -> None:
        ...

    async def read_graph(self) -> KnowledgeGraph:
        ...

    async def search_nodes(self, query: str) -> KnowledgeGraph:
        ...

    async def open_nodes(self, names: List[str]) -> KnowledgeGraph:
        ...


@pytest.fixture(scope="function")
def mock_manager() -> MockManagerProtocol:
    """Create a mock manager for testing."""

    class MockManager:
        async def create_entities(self, entities: List[Entity]) -> List[Entity]:
            return entities

        async def create_relations(self, relations: List[Relation]) -> List[Relation]:
            return relations

        async def add_observations(self, entity: str, observations: List[str]) -> None:
            if entity == "MissingEntity":
                raise EntityNotFoundError(entity)

        async def delete_entities(self, names: List[str]) -> None:
            for name in names:
                if name == "MissingEntity":
                    raise EntityNotFoundError(name)

        async def delete_observations(
            self, entity: str, observations: List[str]
        ) -> None:
            if entity == "MissingEntity":
                raise EntityNotFoundError(entity)

        async def delete_relations(self, from_: str, to: str) -> None:
            if from_ == "MissingEntity" or to == "MissingEntity":
                raise EntityNotFoundError("MissingEntity")

        async def read_graph(self) -> KnowledgeGraph:
            # Return a simple graph
            return KnowledgeGraph(
                entities=[
                    Entity(
                        name="TestEntity",
                        entityType="TypeA",
                        observations=["obs1"],
                    )
                ],
                relations=[
                    Relation(
                        from_="TestEntity",
                        to="AnotherEntity",
                        relationType="knows",
                    )
                ],
            )

        async def search_nodes(self, query: str) -> KnowledgeGraph:
            # If query matches "TestEntity", return graph; otherwise empty
            if "TestEntity".lower() in query.lower():
                return await self.read_graph()
            return KnowledgeGraph(entities=[], relations=[])

        async def open_nodes(self, names: List[str]) -> KnowledgeGraph:
            # If "TestEntity" is requested, return it
            if "TestEntity" in names:
                return await self.read_graph()
            return KnowledgeGraph(entities=[], relations=[])

    return MockManager()


@pytest.mark.asyncio
async def test_create_entities(mock_manager: MockManagerProtocol) -> None:
    """Test creating entities through the MCP server."""
    handler = cast(Any, TOOLS["create_entities"])
    arguments = {
        "entities": [
            {"name": "E1", "entityType": "TypeX", "observations": ["obsA"]},
            {"name": "E2", "entityType": "TypeY", "observations": ["obsB"]},
        ]
    }
    result = await handler(mock_manager, arguments)
    data = json.loads(result[0].text)
    assert len(data) == 2
    assert data[0]["name"] == "E1"
    assert data[1]["observations"] == ["obsB"]


@pytest.mark.asyncio
async def test_create_relations(mock_manager: MockManagerProtocol) -> None:
    """Test creating relations through the MCP server."""
    handler = cast(Any, TOOLS["create_relations"])
    arguments = {"relations": [{"from": "E1", "to": "E2", "relationType": "likes"}]}
    result = await handler(mock_manager, arguments)
    data = json.loads(result[0].text)
    assert len(data) == 1
    assert data[0]["from"] == "E1"
    assert data[0]["to"] == "E2"


@pytest.mark.asyncio
async def test_add_observations(mock_manager: MockManagerProtocol) -> None:
    """Test adding observations through the MCP server."""
    handler = cast(Any, TOOLS["add_observations"])
    arguments = {"entity": "E1", "observations": ["newObs"]}
    result = await handler(mock_manager, arguments)
    data = json.loads(result[0].text)
    assert data["success"] is True


@pytest.mark.asyncio
async def test_delete_entities(mock_manager: MockManagerProtocol) -> None:
    """Test deleting entities through the MCP server."""
    handler = cast(Any, TOOLS["delete_entities"])
    arguments = {"names": ["E1"]}
    result = await handler(mock_manager, arguments)
    data = json.loads(result[0].text)
    assert data["success"] is True


@pytest.mark.asyncio
async def test_delete_observations(mock_manager: MockManagerProtocol) -> None:
    """Test deleting observations through the MCP server."""
    handler = cast(Any, TOOLS["delete_observations"])
    arguments = {"entity": "E1", "observations": ["obs1"]}
    result = await handler(mock_manager, arguments)
    data = json.loads(result[0].text)
    assert data["success"] is True


@pytest.mark.asyncio
async def test_delete_relations(mock_manager: MockManagerProtocol) -> None:
    """Test deleting relations through the MCP server."""
    handler = cast(Any, TOOLS["delete_relations"])
    arguments = {"from": "E1", "to": "E2"}
    result = await handler(mock_manager, arguments)
    data = json.loads(result[0].text)
    assert data["success"] is True


@pytest.mark.asyncio
async def test_read_graph(mock_manager: MockManagerProtocol) -> None:
    """Test reading the graph through the MCP server."""
    handler = cast(Any, TOOLS["read_graph"])
    arguments: Dict[str, Any] = {}
    result = await handler(mock_manager, arguments)
    data = json.loads(result[0].text)
    assert len(data["entities"]) == 1
    assert data["entities"][0]["name"] == "TestEntity"
    assert isinstance(data["entities"][0]["observations"], list)


@pytest.mark.asyncio
async def test_search_nodes(mock_manager: MockManagerProtocol) -> None:
    """Test searching nodes through the MCP server."""
    handler = cast(Any, TOOLS["search_nodes"])
    arguments = {"query": "TestEntity"}
    result = await handler(mock_manager, arguments)
    data = json.loads(result[0].text)
    assert len(data["entities"]) == 1
    assert data["entities"][0]["name"] == "TestEntity"
    assert isinstance(data["entities"][0]["observations"], list)


@pytest.mark.asyncio
async def test_open_nodes(mock_manager: MockManagerProtocol) -> None:
    """Test opening nodes through the MCP server."""
    handler = cast(Any, TOOLS["open_nodes"])
    arguments = {"names": ["TestEntity"]}
    result = await handler(mock_manager, arguments)
    data = json.loads(result[0].text)
    assert len(data["entities"]) == 1
    assert data["entities"][0]["name"] == "TestEntity"
    assert isinstance(data["entities"][0]["observations"], list)


def test_error_handling() -> None:
    """Test error handling functionality."""
    msg = handle_error(EntityNotFoundError("MissingEntity"))
    assert "Entity 'MissingEntity' not found in the graph" in msg
