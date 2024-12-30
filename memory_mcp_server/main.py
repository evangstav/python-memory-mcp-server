#!/usr/bin/env python3
import asyncio
import json
import logging
import argparse
import os
from typing import List, Dict, Any
from urllib.parse import urlparse

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
        "database_url": os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///memory.db"),
        "pool_size": int(os.environ.get("POOL_SIZE", "5")),
        "max_overflow": int(os.environ.get("MAX_OVERFLOW", "10")),
        "pool_timeout": int(os.environ.get("POOL_TIMEOUT", "30")),
        "pool_recycle": int(os.environ.get("POOL_RECYCLE", "3600")),
        "echo": os.environ.get("SQL_ECHO", "").lower() == "true"
    }

def validate_database_url(url: str) -> None:
    """Validate the database URL format."""
    parsed = urlparse(url)
    if parsed.scheme not in ["sqlite+aiosqlite"]:
        raise ValueError(
            "Invalid database URL scheme. Must be 'sqlite+aiosqlite'"
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
        pool_size=config["pool_size"],
        max_overflow=config["max_overflow"],
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
                                    },
                                },
                                "required": ["name", "entityType", "observations"]
                            }
                        }
                    },
                    "required": ["entities"]
                }
            ),
            # Additional tool definitions remain the same
        ]

    @app.call_tool()
    async def call_tool(
        name: str,
        arguments: Dict[str, Any]
    ) -> List[types.TextContent]:
        try:
            result = await getattr(manager, name)(**arguments)
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result) if result is not None else "Operation completed successfully"
                )
            ]
        except Exception as e:
            error_message = f"Error in {name}: {str(e)}"
            logger.error(error_message, exc_info=True)
            return [types.TextContent(type="text", text=f"Error: {error_message}")]

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