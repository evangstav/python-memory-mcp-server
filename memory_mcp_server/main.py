#!/usr/bin/env python3
import asyncio
import json
import logging
import argparse
import os
import time
from typing import List, Dict, Any
from urllib.parse import urlparse
from asyncio import Semaphore
from functools import wraps

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .optimized_sqlite_manager import OptimizedSQLiteManager
from .exceptions import (
    KnowledgeGraphError,
    EntityNotFoundError,
    EntityAlreadyExistsError,
    RelationValidationError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("knowledge-graph-server")

def parse_database_config() -> Dict[str, Any]:
    """Parse database configuration from environment variables."""
    return {
        "database_url": os.environ.get("DATABASE_URL", "sqlite:///memory.db"),
        "echo": os.environ.get("SQL_ECHO", "").lower() == "true"
    }

def validate_database_url(url: str) -> None:
    """Validate the database URL format."""
    parsed = urlparse(url)
    if parsed.scheme not in ["sqlite"]:
        raise ValueError(
            "Invalid database URL scheme. Must be 'sqlite'"
        )

async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-url",
        type=str,
        help="SQLite database URL (e.g., sqlite+aiosqlite:///path/to/db)"
    )
    args = parser.parse_args()

    # Get database configuration
    config = parse_database_config()
    if args.database_url:
        config["database_url"] = args.database_url

    # Validate database URL
    validate_database_url(config["database_url"])

    # Initialize the optimized SQLite manager
    manager = OptimizedSQLiteManager(
        database_url=config["database_url"],
        echo=config["echo"]
    )

    # Initialize database schema and indices
    await manager.initialize()
    
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
                                    "name": {"type": "string"},
                                    "entityType": {"type": "string"},
                                    "observations": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["name", "entityType", "observations"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["entities"],
                    "additionalProperties": False
                }
            ),
            types.Tool(
                name="create_relations",
                description="Create multiple new relations between entities",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                    "relationType": {"type": "string"}
                                },
                                "required": ["from", "to", "relationType"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["relations"],
                    "additionalProperties": False
                }
            ),
            types.Tool(
                name="read_graph",
                description="Read the entire knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            ),
            types.Tool(
                name="search_nodes",
                description="Search for nodes based on query",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            ),
            types.Tool(
                name="add_observations",
                description="Add new observations to existing entities",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "observations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entityName": {"type": "string"},
                                    "contents": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["entityName", "contents"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["observations"],
                    "additionalProperties": False
                }
            ),
            types.Tool(
                name="delete_entities",
                description="Remove entities and their relations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entityNames": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["entityNames"],
                    "additionalProperties": False
                }
            ),
            types.Tool(
                name="delete_observations",
                description="Remove specific observations from entities",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "deletions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entityName": {"type": "string"},
                                    "observations": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["entityName", "observations"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["deletions"],
                    "additionalProperties": False
                }
            ),
            types.Tool(
                name="delete_relations",
                description="Remove specific relations from the graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                    "relationType": {"type": "string"}
                                },
                                "required": ["from", "to", "relationType"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["relations"],
                    "additionalProperties": False
                }
            ),
            types.Tool(
                name="open_nodes",
                description="Retrieve specific nodes by name",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["names"],
                    "additionalProperties": False
                }
            )
        ]

    def rate_limit(max_requests: int = 100, window_seconds: int = 60):
        semaphore = Semaphore(max_requests)
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                async with semaphore:
                    return await func(*args, **kwargs)
            return wrapper
        return decorator

    async def health_check() -> Dict[str, Any]:
        """Check the health of the server and database."""
        try:
            await manager.read_graph()
            return {"status": "healthy", "timestamp": time.time()}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    @app.call_tool()
    @rate_limit()
    async def call_tool(
        name: str,
        arguments: Dict[str, Any]
    ) -> List[types.TextContent]:
        try:
            result = await getattr(manager, name)(**arguments)
            return [types.TextContent(
                type="text",
                text=json.dumps(result) if result is not None else "Operation completed successfully"
            )]
        except Exception as e:
            error_message = f"Error in {name}: {str(e)}"
            logger.error(error_message, exc_info=True)
            return [types.TextContent(type="text", text=error_message)]

    async with stdio_server() as (read_stream, write_stream):
        logger.info(
            f"Knowledge Graph MCP Server running on stdio "
            f"(database: {config['database_url']})"
        )
        try:
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
        finally:
            await manager.cleanup()

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)
