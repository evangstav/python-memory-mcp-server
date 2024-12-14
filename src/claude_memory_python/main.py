#!/usr/bin/env python3
import asyncio
import json
import logging
import argparse
from pathlib import Path
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
        "--path",
        type=Path,
        default=Path(__file__).parent / "memory.jsonl",
        help="Path to the memory file",
    )
    args = parser.parse_args()

    # Create manager instance
    manager = KnowledgeGraphManager(args.path)