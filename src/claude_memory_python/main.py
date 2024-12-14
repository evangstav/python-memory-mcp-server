#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import argparse
from typing import List, Dict, Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .knowledge_graph_manager import KnowledgeGraphManager
from .interfaces import Relation, Entity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knowledge-graph-server")


async def async_main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory-path",
        default=os.path.join(os.path.dirname(__file__), "memory.jsonl"),
        help="Path to the memory file",
    )
    args = parser.parse_args()

    # Create manager instance
    manager = KnowledgeGraphManager(args.memory_path)

    # Create server instance
    app = Server("knowledge-graph-server")

    @app.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="create_entities",
                description="Create multiple new entities in the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The name of the entity",
                                    },
                                    "entityType": {
                                        "type": "string",
                                        "description": "The type of the entity",
                                    },
                                    "observations": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "An array of observation contents",
                                    },
                                },
                                "required": ["name", "entityType", "observations"],
                            },
                        }
                    },
                    "required": ["entities"],
                },
            ),
            types.Tool(
                name="create_relations",
                description="Create multiple new relations between entities in the knowledge graph. Relations should be in active voice",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {
                                        "type": "string",
                                        "description": "The name of the entity where the relation starts",
                                    },
                                    "to": {
                                        "type": "string",
                                        "description": "The name of the entity where the relation ends",
                                    },
                                    "relationType": {
                                        "type": "string",
                                        "description": "The type of the relation",
                                    },
                                },
                                "required": ["from", "to", "relationType"],
                            },
                        }
                    },
                    "required": ["relations"],
                },
            ),
            types.Tool(
                name="add_observations",
                description="Add new observations to existing entities in the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "observations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entityName": {
                                        "type": "string",
                                        "description": "The name of the entity to add the observations to",
                                    },
                                    "contents": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "An array of observation contents to add",
                                    },
                                },
                                "required": ["entityName", "contents"],
                            },
                        }
                    },
                    "required": ["observations"],
                },
            ),
            types.Tool(
                name="delete_entities",
                description="Delete multiple entities and their associated relations from the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entityNames": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "An array of entity names to delete",
                        }
                    },
                    "required": ["entityNames"],
                },
            ),
            types.Tool(
                name="delete_observations",
                description="Delete specific observations from entities in the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "deletions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entityName": {
                                        "type": "string",
                                        "description": "The name of the entity containing the observations",
                                    },
                                    "observations": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "An array of observations to delete",
                                    },
                                },
                                "required": ["entityName", "observations"],
                            },
                        }
                    },
                    "required": ["deletions"],
                },
            ),
            types.Tool(
                name="delete_relations",
                description="Delete multiple relations from the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {
                                        "type": "string",
                                        "description": "The name of the entity where the relation starts",
                                    },
                                    "to": {
                                        "type": "string",
                                        "description": "The name of the entity where the relation ends",
                                    },
                                    "relationType": {
                                        "type": "string",
                                        "description": "The type of the relation",
                                    },
                                },
                                "required": ["from", "to", "relationType"],
                            },
                        }
                    },
                    "required": ["relations"],
                },
            ),
            types.Tool(
                name="read_graph",
                description="Read the entire knowledge graph",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="search_nodes",
                description="Search for nodes in the knowledge graph based on a query",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to match against entity names, types, and observation content",
                        }
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="open_nodes",
                description="Open specific nodes in the knowledge graph by their names",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "An array of entity names to retrieve",
                        }
                    },
                    "required": ["names"],
                },
            ),
        ]

    @app.call_tool()
    async def call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[types.TextContent]:
        try:
            if name == "create_entities":
                entities = [
                    Entity(
                        name=e["name"],
                        entityType=e["entityType"],
                        observations=e["observations"],
                    )
                    for e in arguments["entities"]
                ]
                result = await manager.create_entities(entities)
                return [
                    types.TextContent(
                        type="text", text=json.dumps(result, default=vars)
                    )
                ]

            elif name == "create_relations":
                relations = [
                    Relation(
                        from_=r["from"], to=r["to"], relationType=r["relationType"]
                    )
                    for r in arguments["relations"]
                ]
                result = await manager.create_relations(relations)
                return [
                    types.TextContent(
                        type="text", text=json.dumps(result, default=vars)
                    )
                ]

            elif name == "add_observations":
                result = await manager.add_observations(arguments["observations"])
                return [types.TextContent(type="text", text=json.dumps(result))]

            elif name == "delete_entities":
                await manager.delete_entities(arguments["entityNames"])
                return [
                    types.TextContent(type="text", text="Entities deleted successfully")
                ]

            elif name == "delete_observations":
                await manager.delete_observations(arguments["deletions"])
                return [
                    types.TextContent(
                        type="text", text="Observations deleted successfully"
                    )
                ]

            elif name == "delete_relations":
                relations = [
                    Relation(
                        from_=r["from"], to=r["to"], relationType=r["relationType"]
                    )
                    for r in arguments["relations"]
                ]
                await manager.delete_relations(relations)
                return [
                    types.TextContent(
                        type="text", text="Relations deleted successfully"
                    )
                ]

            elif name == "read_graph":
                graph = await manager.read_graph()
                return [
                    types.TextContent(type="text", text=json.dumps(graph, default=vars))
                ]

            elif name == "search_nodes":
                result = await manager.search_nodes(arguments["query"])
                return [
                    types.TextContent(
                        type="text", text=json.dumps(result, default=vars)
                    )
                ]

            elif name == "open_nodes":
                result = await manager.open_nodes(arguments["names"])
                return [
                    types.TextContent(
                        type="text", text=json.dumps(result, default=vars)
                    )
                ]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.error(f"Error in tool {name}: {str(e)}", exc_info=True)
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Knowledge Graph MCP Server running on stdio")
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()