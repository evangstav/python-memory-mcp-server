#!/usr/bin/env python3
"""MCP server implementation for memory management."""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import StdioServerTransport
from mcp.types import (
    CallToolRequestSchema,
    ErrorCode,
    ListToolsRequestSchema,
    McpError,
)

from .interfaces import Entity, Relation
from .knowledge_graph_manager import KnowledgeGraphManager

logger = logging.getLogger(__name__)


def parse_arguments() -> Dict[str, Any]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Memory MCP Server")
    parser.add_argument(
        "--path",
        type=str,
        default="memory.jsonl",
        help="Path to memory file (default: memory.jsonl)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=60,
        help="Cache TTL in seconds (default: 60)",
    )
    return vars(parser.parse_args())


class MemoryServer:
    """MCP server implementation for memory management."""

    def __init__(self, memory_file: Path, cache_ttl: int = 60):
        """Initialize the memory server."""
        self.server = Server(
            {
                "name": "memory-mcp-server",
                "version": "0.1.0",
            },
            {
                "capabilities": {
                    "tools": {},
                },
            },
        )

        self.knowledge_graph = KnowledgeGraphManager(memory_file, cache_ttl)
        self.setup_handlers()

    def setup_handlers(self) -> None:
        """Set up request handlers."""
        self.server.setRequestHandler(ListToolsRequestSchema, self.list_tools)
        self.server.setRequestHandler(CallToolRequestSchema, self.call_tool)

    async def create_entities(self, entities: List[Entity]) -> Dict[str, Any]:
        """Create multiple new entities.

        Args:
            entities: List of entities to create

        Returns:
            Dict containing success status
        """
        try:
            await self.knowledge_graph.create_entities(entities)
            return {"success": True}
        except Exception as err:
            raise McpError(ErrorCode.InternalError, str(err)) from err

    async def create_relations(self, relations: List[Relation]) -> Dict[str, Any]:
        """Create multiple new relations.

        Args:
            relations: List of relations to create

        Returns:
            Dict containing success status
        """
        try:
            await self.knowledge_graph.create_relations(relations)
            return {"success": True}
        except Exception as err:
            raise McpError(ErrorCode.InternalError, str(err)) from err

    async def add_observations(
        self, entity: str, observations: List[str]
    ) -> Dict[str, Any]:
        """Add observations to an existing entity.

        Args:
            entity: Name of the entity to add observations to
            observations: List of observations to add

        Returns:
            Dict containing success status
        """
        try:
            await self.knowledge_graph.add_observations(entity, observations)
            return {"success": True}
        except Exception as err:
            raise McpError(ErrorCode.InternalError, str(err)) from err

    async def delete_entities(self, names: List[str]) -> Dict[str, Any]:
        """Delete entities and their relations.

        Args:
            names: List of entity names to delete

        Returns:
            Dict containing success status
        """
        try:
            # TODO: Implement delete_entities in KnowledgeGraphManager
            return {"success": True}
        except Exception as err:
            raise McpError(ErrorCode.InternalError, str(err)) from err

    async def delete_observations(
        self, entity: str, observations: List[str]
    ) -> Dict[str, Any]:
        """Delete specific observations from an entity.

        Args:
            entity: Name of the entity to delete observations from
            observations: List of observations to delete

        Returns:
            Dict containing success status
        """
        try:
            # TODO: Implement delete_observations in KnowledgeGraphManager
            return {"success": True}
        except Exception as err:
            raise McpError(ErrorCode.InternalError, str(err)) from err

    async def delete_relations(self, from_: str, to: str) -> Dict[str, Any]:
        """Delete relations between entities.

        Args:
            from_: Source entity name
            to: Target entity name

        Returns:
            Dict containing success status
        """
        try:
            # TODO: Implement delete_relations in KnowledgeGraphManager
            return {"success": True}
        except Exception as err:
            raise McpError(ErrorCode.InternalError, str(err)) from err

    async def open_nodes(self, names: List[str]) -> Dict[str, Any]:
        """Retrieve specific nodes by name.

        Args:
            names: List of entity names to retrieve

        Returns:
            Dict containing the requested nodes
        """
        try:
            # TODO: Implement open_nodes in KnowledgeGraphManager
            return {"nodes": []}
        except Exception as err:
            raise McpError(ErrorCode.InternalError, str(err)) from err

    async def read_graph(self) -> Dict[str, Any]:
        """Read the entire knowledge graph."""
        try:
            graph = await self.knowledge_graph.read_graph()
            return {"graph": graph.to_dict()}
        except Exception as err:
            raise McpError(ErrorCode.InternalError, str(err)) from err

    async def search_nodes(self, query: str) -> Dict[str, Any]:
        """Search nodes by query."""
        try:
            results = await self.knowledge_graph.search_nodes(query)
            return {"results": results.to_dict()}
        except Exception as err:
            raise McpError(ErrorCode.InternalError, str(err)) from err

    def parse_entities(self, data: List[Dict[str, Any]]) -> List[Entity]:
        """Parse entity data into Entity objects."""
        return [
            Entity(
                name=item["name"],
                entityType=item["entityType"],
                observations=item.get("observations", []),
            )
            for item in data
        ]

    def parse_relations(self, data: List[Dict[str, Any]]) -> List[Relation]:
        """Parse relation data into Relation objects."""
        return [
            Relation(
                from_=item["from"],
                to=item["to"],
                relationType=item["relationType"],
            )
            for item in data
        ]

    async def list_tools(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """List available tools."""
        return {
            "tools": [
                {
                    "name": "create_entities",
                    "description": (
                        "Create multiple new entities in the knowledge graph"
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "entityType": {"type": "string"},
                                        "observations": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["name", "entityType"],
                                },
                            },
                        },
                        "required": ["entities"],
                    },
                },
                {
                    "name": "create_relations",
                    "description": (
                        "Create relations between entities (in active voice)"
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "relations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "from": {"type": "string"},
                                        "to": {"type": "string"},
                                        "relationType": {"type": "string"},
                                    },
                                    "required": ["from", "to", "relationType"],
                                },
                            },
                        },
                        "required": ["relations"],
                    },
                },
                {
                    "name": "add_observations",
                    "description": "Add new observations to existing entities",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string"},
                            "observations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["entity", "observations"],
                    },
                },
                {
                    "name": "delete_entities",
                    "description": "Delete entities and their relations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "names": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["names"],
                    },
                },
                {
                    "name": "delete_observations",
                    "description": "Delete specific observations from entities",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string"},
                            "observations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["entity", "observations"],
                    },
                },
                {
                    "name": "delete_relations",
                    "description": "Delete specific relations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                        },
                    },
                },
                {
                    "name": "read_graph",
                    "description": "Read the entire knowledge graph",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "search_nodes",
                    "description": "Search entities and relations by query",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "open_nodes",
                    "description": "Retrieve specific nodes by name",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "names": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["names"],
                    },
                },
            ]
        }

    async def call_tool(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool calls.

        Args:
            request: Tool request containing name and arguments

        Returns:
            Dict containing the tool's response

        Raises:
            McpError: If tool is not found or execution fails
        """
        tool_name = request["params"]["name"]
        arguments = request["params"]["arguments"]
        result: Dict[str, Any]

        if tool_name == "create_entities":
            entities = self.parse_entities(arguments["entities"])
            await self.knowledge_graph.create_entities(entities)
            result = {"success": True}

        elif tool_name == "create_relations":
            relations = self.parse_relations(arguments["relations"])
            await self.knowledge_graph.create_relations(relations)
            result = {"success": True}

        elif tool_name == "add_observations":
            result = await self.add_observations(
                arguments["entity"],
                arguments["observations"],
            )

        elif tool_name == "delete_entities":
            result = await self.delete_entities(arguments["names"])

        elif tool_name == "delete_observations":
            result = await self.delete_observations(
                arguments["entity"],
                arguments["observations"],
            )

        elif tool_name == "delete_relations":
            result = await self.delete_relations(
                arguments.get("from"),
                arguments.get("to"),
            )

        elif tool_name == "read_graph":
            result = await self.read_graph()

        elif tool_name == "search_nodes":
            result = await self.search_nodes(arguments["query"])

        elif tool_name == "open_nodes":
            result = await self.open_nodes(arguments["names"])

        else:
            raise McpError(ErrorCode.MethodNotFound, f"Unknown tool: {tool_name}")

        return result

    async def run(self) -> None:
        """Run the server."""
        await self.knowledge_graph.initialize()
        transport = StdioServerTransport()
        await self.server.connect(transport)
        logger.info("Memory MCP server running on stdio")


def main() -> None:
    """Run the server."""
    args = parse_arguments()
    server = MemoryServer(args["path"], args["cache_ttl"])
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
