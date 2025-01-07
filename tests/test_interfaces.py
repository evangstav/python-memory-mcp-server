"""Tests for interface classes."""

from memory_mcp_server.interfaces import Entity, KnowledgeGraph, Relation


def test_entity_creation() -> None:
    """Test entity creation and attributes."""
    entity = Entity(
        name="TestEntity", entityType="TestType", observations=["obs1", "obs2"]
    )
    assert entity.name == "TestEntity"
    assert entity.entityType == "TestType"
    assert len(entity.observations) == 2
    assert "obs1" in entity.observations
    assert "obs2" in entity.observations


def test_relation_creation() -> None:
    """Test relation creation and attributes."""
    relation = Relation(from_="EntityA", to="EntityB", relationType="TestRelation")
    assert relation.from_ == "EntityA"
    assert relation.to == "EntityB"
    assert relation.relationType == "TestRelation"


def test_knowledge_graph_creation() -> None:
    """Test knowledge graph creation and attributes."""
    entities = [
        Entity(name="E1", entityType="T1", observations=[]),
        Entity(name="E2", entityType="T2", observations=[]),
    ]
    relations = [Relation(from_="E1", to="E2", relationType="R1")]
    graph = KnowledgeGraph(entities=entities, relations=relations)
    assert len(graph.entities) == 2
    assert len(graph.relations) == 1
    assert graph.entities[0].name == "E1"
    assert graph.relations[0].from_ == "E1"
