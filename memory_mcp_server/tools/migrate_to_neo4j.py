#!/usr/bin/env python3
"""Migration tool to transfer data from JSONL to Neo4j backend."""

import asyncio
import argparse
import os
from pathlib import Path

from ..backends.jsonl import JsonlBackend
from ..backends.neo4j import Neo4jBackend


async def migrate_data(
    jsonl_path: Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    batch_size: int = 100
) -> None:
    """Migrate data from JSONL to Neo4j.
    
    Args:
        jsonl_path: Path to JSONL file
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        batch_size: Number of entities/relations to migrate in each batch
    """
    # Initialize backends
    jsonl_backend = JsonlBackend(jsonl_path)
    neo4j_backend = Neo4jBackend(neo4j_uri, neo4j_user, neo4j_password)

    try:
        # Initialize both backends
        await jsonl_backend.initialize()
        await neo4j_backend.initialize()

        # Read entire graph from JSONL
        print(f"Reading data from {jsonl_path}...")
        graph = await jsonl_backend.read_graph()
        
        total_entities = len(graph.entities)
        total_relations = len(graph.relations)
        print(f"Found {total_entities} entities and {total_relations} relations")

        # Migrate entities in batches
        print("\nMigrating entities...")
        for i in range(0, total_entities, batch_size):
            batch = graph.entities[i:i + batch_size]
            await neo4j_backend.create_entities(batch)
            print(f"Migrated entities {i + 1}-{min(i + batch_size, total_entities)}")

        # Migrate relations in batches
        print("\nMigrating relations...")
        for i in range(0, total_relations, batch_size):
            batch = graph.relations[i:i + batch_size]
            await neo4j_backend.create_relations(batch)
            print(f"Migrated relations {i + 1}-{min(i + batch_size, total_relations)}")

        print("\nMigration completed successfully!")

    finally:
        # Ensure both backends are properly closed
        await jsonl_backend.close()
        await neo4j_backend.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate data from JSONL to Neo4j")
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        required=True,
        help="Path to source JSONL file"
    )
    parser.add_argument(
        "--neo4j-uri",
        help="Neo4j connection URI (can also use NEO4J_URI env var)",
    )
    parser.add_argument(
        "--neo4j-user",
        help="Neo4j username (can also use NEO4J_USER env var)",
    )
    parser.add_argument(
        "--neo4j-password",
        help="Neo4j password (can also use NEO4J_PASSWORD env var)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of items to migrate in each batch (default: 100)"
    )

    args = parser.parse_args()

    # Get Neo4j configuration from args or environment
    neo4j_uri = args.neo4j_uri or os.getenv("NEO4J_URI")
    neo4j_user = args.neo4j_user or os.getenv("NEO4J_USER")
    neo4j_password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        parser.error(
            "Neo4j configuration required. Provide either command line arguments "
            "or environment variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)"
        )

    try:
        asyncio.run(migrate_data(
            jsonl_path=args.jsonl_path,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            batch_size=args.batch_size
        ))
    except KeyboardInterrupt:
        print("\nMigration interrupted by user")
    except Exception as e:
        print(f"\nError during migration: {e}")
        exit(1)


if __name__ == "__main__":
    main()