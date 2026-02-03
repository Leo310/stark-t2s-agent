"""Build PostgreSQL, Fuseki, and Qdrant stores from STaRK-Prime data.

This module provides the entry point for the build-prime-stores CLI command.
"""

import argparse
import sys

from stark_prime_t2s.config import (
    FUSEKI_QUERY_URL,
    POSTGRES_URL,
    QDRANT_COLLECTION,
    QDRANT_COLLECTION_FULL,
    QDRANT_HOST,
    QDRANT_PORT,
)
from stark_prime_t2s.dataset.download_prime import download_prime_skb
from stark_prime_t2s.dataset.parse_prime_processed import PrimeDataLoader


def _extract_rich_metadata(info: dict, node_type: str = "") -> dict:
    """Extract rich metadata from node_info for embedding.

    Embeds all non-empty descriptive fields from the details dictionary,
    excluding technical ID fields that don't add semantic meaning.
    """
    import math

    name = str(info.get("name", "") or "")

    # Extract from details if available
    details = info.get("details", {}) or {}

    # Handle full_name - can be string, list, or None
    raw_full_name = details.get("name", "")
    if isinstance(raw_full_name, list):
        full_name = ", ".join(str(x) for x in raw_full_name) if raw_full_name else ""
    else:
        full_name = str(raw_full_name or "")

    aliases = details.get("alias", []) or []

    # Build rich description from all descriptive fields in details
    rich_parts = []

    # Add full name if different from short name
    if full_name and full_name.lower() != name.lower():
        rich_parts.append(full_name)

    # Add aliases
    if aliases and isinstance(aliases, list):
        alias_str = ", ".join(str(a) for a in aliases[:10])  # Limit to 10 aliases
        if alias_str:
            rich_parts.append(f"Also known as: {alias_str}")

    # Fields to exclude (technical IDs that don't add semantic meaning)
    exclude_fields = {
        # ID fields
        "_id",
        "_score",
        "query",
        "mondo_id",
        "dbId",
        "stId",
        "stIdVersion",
        "group_id_bert",
        # Boolean/technical flags
        "hasDiagram",
        "hasEHLD",
        "hasEvent",
        "isInDisease",
        "isInferred",
        # Empty/unused fields
        "notfound",
        "previousReviewStatus",
        "releaseStatus",
        "normalPathway",
    }

    # Add all descriptive fields from details (excluding IDs)
    for key, value in details.items():
        if key in exclude_fields:
            continue

        # Skip None
        if value is None:
            continue
        # Skip float NaN
        if isinstance(value, float) and math.isnan(value):
            continue
        # Skip string placeholders
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in ("nan", "none", "", "null", "n/a"):
                continue

        # Format the value based on type
        if isinstance(value, list):
            if len(value) == 0:
                continue
            # For lists, join string representations
            if all(isinstance(v, str) for v in value):
                value_str = ", ".join(value)
            else:
                # For list of dicts/objects, try to extract meaningful text
                value_str = _format_complex_value(value)
        elif isinstance(value, dict):
            value_str = _format_complex_value(value)
        else:
            value_str = str(value)

        if value_str and len(value_str.strip()) > 0:
            # Add with field name for context
            rich_parts.append(f"{key}: {value_str}")

    return {
        "name": name,
        "full_name": full_name,
        "description": "\n".join(rich_parts) if rich_parts else "",
    }


def _format_complex_value(value) -> str:
    """Format complex values (dicts, lists) into readable text."""
    if isinstance(value, dict):
        # For dicts, extract text fields or convert to readable format
        if "displayName" in value:
            return str(value["displayName"])
        elif "name" in value:
            return str(value["name"])
        elif "title" in value:
            return str(value["title"])
        else:
            # Convert to simple key: value format
            parts = []
            for k, v in value.items():
                if isinstance(v, (str, int, float)):
                    parts.append(f"{k}: {v}")
            return ", ".join(parts) if parts else ""

    elif isinstance(value, list):
        # Recursively format list items
        formatted = []
        for item in value:
            if isinstance(item, (str, int, float)):
                formatted.append(str(item))
            elif isinstance(item, dict):
                formatted.append(_format_complex_value(item))
        return ", ".join(formatted) if formatted else ""

    else:
        return str(value)


def build_qdrant_index(
    loader: PrimeDataLoader,
    force: bool = False,
    collection_name: str | None = None,
):
    """Build the Qdrant vector index for entity search.

    Extracts rich metadata (full names, aliases, summaries) from the STaRK-Prime
    node_info to enable better semantic search.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from langchain_openai import OpenAIEmbeddings
    from stark_prime_t2s.config import OPENAI_API_KEY

    print("Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    target_collection = collection_name or QDRANT_COLLECTION

    # Check if collection exists and determine resume point
    resume_from: int = 0
    collection_exists = False

    try:
        collections = client.get_collections().collections
        collection_exists = any(c.name == target_collection for c in collections)

        if collection_exists:
            info = client.get_collection(target_collection)
            current_count = info.points_count or 0
            if current_count > 0:
                if not force:
                    # Resume mode: start from where we left off
                    resume_from = current_count
                    print(
                        f"  → Collection exists with {resume_from} entities, resuming from entity {resume_from}"
                    )
                else:
                    # Force mode: delete and restart
                    print(
                        f"  Deleting existing collection '{target_collection}' ({current_count} entities)..."
                    )
                    client.delete_collection(target_collection)
                    collection_exists = False
    except Exception as e:
        print(f"  Warning checking Qdrant: {e}")

    # Get entities from loader with rich metadata
    print("  Loading entities with rich metadata...")
    entities: list[dict[str, object]] = []
    entities_with_rich_desc = 0

    for node_id in range(loader.num_nodes):
        info = loader.node_info.get(node_id, {})
        node_type_id = int(loader.node_types[node_id].item())
        node_type = loader.node_type_dict.get(node_type_id, "unknown")

        # Extract rich metadata
        metadata = _extract_rich_metadata(info, node_type)

        if metadata["description"]:
            entities_with_rich_desc += 1

        entities.append(
            {
                "id": node_id,
                "type": node_type,
                "name": metadata["name"] or f"Entity {node_id}",
                "full_name": metadata["full_name"],
                "description": metadata["description"],
            }
        )

    print(f"  Found {entities_with_rich_desc}/{len(entities)} entities with rich descriptions")

    # Skip already processed entities if resuming
    if resume_from > 0:
        if resume_from >= len(entities):
            print(f"  ✓ All {len(entities)} entities already indexed, nothing to do")
            return
        entities = entities[resume_from:]
        print(f"  Resuming: will process {len(entities)} remaining entities")

    # Get dimension by embedding a single test vector (if creating new collection)
    print("  Getting embedding dimension...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
    test_vector = embeddings.embed_query("test")
    dimension = len(test_vector)
    print(f"  Embedding dimension: {dimension}")

    # Create collection if it doesn't exist
    if not collection_exists:
        print(f"  Creating collection...")
        client.create_collection(
            collection_name=target_collection,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )
    else:
        print(f"  Using existing collection...")

    # Process in batches: embed and upload incrementally
    # Batch size reduced to 500 to stay under Qdrant's 32MB payload limit
    # Each entity: ~6KB vector + ~1-2KB payload = ~7-8KB per entity
    # 500 entities × 8KB = ~4MB per batch (well under 32MB limit)
    batch_size = 500
    total_uploaded = 0
    print(f"  Embedding and uploading {len(entities)} entities in batches of {batch_size}...")

    from tqdm import tqdm

    # Create progress bar for total entities
    with tqdm(total=len(entities), desc="  Uploading to Qdrant", unit="entities") as pbar:
        for i in range(0, len(entities), batch_size):
            batch_entities = entities[i : i + batch_size]

            # Build texts for this batch
            texts = []
            for e in batch_entities:
                parts = [f"{e['type']}: {e['name']}"]
                if e["full_name"]:
                    parts.append(e["full_name"])
                if e["description"]:
                    parts.append(str(e["description"]))
                texts.append("\n".join(parts))

            # Embed this batch
            vectors = embeddings.embed_documents(texts)

            # Create points
            points = []
            for j, (entity, vector) in enumerate(zip(batch_entities, vectors)):
                points.append(
                    PointStruct(
                        id=resume_from + i + j,
                        vector=vector,
                        payload={
                            "entity_id": entity["id"],
                            "type": entity["type"],
                            "name": entity["name"],
                            "full_name": entity.get("full_name", ""),
                            "description": entity.get("description", ""),
                        },
                    )
                )

            # Upload immediately
            client.upsert(collection_name=target_collection, points=points)
            total_uploaded += len(points)
            pbar.update(len(points))

    print(f"  ✓ Qdrant index built with {total_uploaded} entities")


def main():
    """Main entry point for build-prime-stores command."""
    parser = argparse.ArgumentParser(
        description="Build PostgreSQL, Fuseki, and Qdrant stores from STaRK-Prime data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and rebuild even if data exists",
    )
    parser.add_argument(
        "--skip-rdf",
        action="store_true",
        help="Skip building the RDF graph (Fuseki)",
    )
    parser.add_argument(
        "--skip-sql",
        action="store_true",
        help="Skip building the SQL database (PostgreSQL)",
    )
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Skip building the Qdrant vector index",
    )
    parser.add_argument(
        "--qdrant-collection",
        type=str,
        default=None,
        help=(
            "Target Qdrant collection name for the entity index. "
            "Defaults to QDRANT_COLLECTION or QDRANT_COLLECTION_FULL when --qdrant-full is set."
        ),
    )
    parser.add_argument(
        "--qdrant-full",
        action="store_true",
        help=(
            "Build a full-description Qdrant index in QDRANT_COLLECTION_FULL. "
            "Does not overwrite the default collection unless --qdrant-collection is set."
        ),
    )

    args = parser.parse_args()

    print("=" * 60)
    print("STaRK-Prime Data Store Builder")
    print("=" * 60)
    print()
    print("Docker services:")
    print(f"  PostgreSQL: {POSTGRES_URL.split('@')[1] if '@' in POSTGRES_URL else POSTGRES_URL}")
    print(f"  Fuseki: {FUSEKI_QUERY_URL}")
    print(f"  Qdrant: {QDRANT_HOST}:{QDRANT_PORT}/{QDRANT_COLLECTION}")
    print(f"  Qdrant (full): {QDRANT_HOST}:{QDRANT_PORT}/{QDRANT_COLLECTION_FULL}")
    print()

    # Download data
    print("Step 1: Download STaRK-Prime data")
    print("-" * 40)
    download_prime_skb(force=args.force)
    print()

    # Load data
    print("Step 2: Load and parse data")
    print("-" * 40)
    loader = PrimeDataLoader()
    loader.print_stats()
    print()

    # Build PostgreSQL
    if not args.skip_sql:
        print("Step 3: Build PostgreSQL database")
        print("-" * 40)
        from stark_prime_t2s.materialize.postgres_prime import PostgresPrimeStore

        try:
            store = PostgresPrimeStore()
            if not args.force and store.is_available():
                print("  → PostgreSQL already has data")
                print("  → Use --force to rebuild")
            else:
                store.build_from_loader(loader)
        except Exception as e:
            print(f"  ERROR: Could not connect to PostgreSQL: {e}")
            print("  Make sure Docker containers are running: docker-compose up -d")
            sys.exit(1)
        print()

    # Build Fuseki
    if not args.skip_rdf:
        print("Step 4: Build Fuseki RDF graph")
        print("-" * 40)
        from stark_prime_t2s.materialize.fuseki_prime import FusekiPrimeStore

        try:
            store = FusekiPrimeStore()
            if not args.force and store.is_available():
                print("  → Fuseki already has data")
                print("  → Use --force to rebuild")
            else:
                store.build_from_loader(loader)
        except Exception as e:
            print(f"  ERROR: Could not connect to Fuseki: {e}")
            print("  Make sure Docker containers are running: docker-compose up -d")
            sys.exit(1)
        print()

    # Build Qdrant
    if not args.skip_qdrant:
        print("Step 5: Build Qdrant vector index")
        print("-" * 40)
        try:
            collection_name = args.qdrant_collection
            if args.qdrant_full and not collection_name:
                collection_name = QDRANT_COLLECTION_FULL
            build_qdrant_index(loader, force=args.force, collection_name=collection_name)
        except Exception as e:
            print(f"  ERROR: Could not build Qdrant index: {e}")
            print("  Make sure Qdrant is running: docker-compose up -d qdrant")
            sys.exit(1)
        print()

    print("=" * 60)
    print("Build complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Make sure OPENAI_API_KEY is set in your .env file")
    print("  2. Run the demo: python scripts/demo_chat.py")
    print("  3. Run the benchmark: python -m stark_prime_t2s.benchmark.run_prime")


if __name__ == "__main__":
    main()
