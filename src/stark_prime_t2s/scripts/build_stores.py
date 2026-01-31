"""Build PostgreSQL, Fuseki, and Qdrant stores from STaRK-Prime data.

This module provides the entry point for the build-prime-stores CLI command.
"""

import argparse
import sys

from stark_prime_t2s.config import (
    FUSEKI_QUERY_URL,
    POSTGRES_URL,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from stark_prime_t2s.data.download_prime import download_prime_skb
from stark_prime_t2s.data.parse_prime_processed import PrimeDataLoader


def _extract_rich_metadata(info: dict) -> dict:
    """Extract rich metadata from node_info, including details from raw data.
    
    The STaRK-Prime dataset stores additional info in a 'details' sub-dict:
    - details.name: Full name (e.g., "endothelin receptor type B" vs symbol "EDNRB")
    - details.alias: List of synonyms/aliases
    - details.summary: Description/summary text
    """
    name = str(info.get("name", "") or "")
    description = str(info.get("description", "") or "")
    
    # Extract from details if available
    details = info.get("details", {}) or {}
    
    # Handle full_name - can be string, list, or None
    raw_full_name = details.get("name", "")
    if isinstance(raw_full_name, list):
        full_name = ", ".join(str(x) for x in raw_full_name) if raw_full_name else ""
    else:
        full_name = str(raw_full_name or "")
    
    aliases = details.get("alias", []) or []
    
    # Handle summary - can be string, list, or None
    raw_summary = details.get("summary", "")
    if isinstance(raw_summary, list):
        summary = " ".join(str(x) for x in raw_summary) if raw_summary else ""
    else:
        summary = str(raw_summary or "")
    
    # Build rich description for embedding
    rich_parts = []
    
    # Add full name if different from short name
    if full_name and full_name.lower() != name.lower():
        rich_parts.append(full_name)
    
    # Add aliases
    if aliases and isinstance(aliases, list):
        alias_str = ", ".join(str(a) for a in aliases[:10])  # Limit to 10 aliases
        if alias_str:
            rich_parts.append(f"Also known as: {alias_str}")
    
    # Add summary/description
    if summary:
        rich_parts.append(summary[:500])
    elif description:
        rich_parts.append(description[:500])
    
    return {
        "name": name,
        "full_name": full_name,
        "description": "\n".join(rich_parts) if rich_parts else "",
    }


def build_qdrant_index(loader: PrimeDataLoader, force: bool = False):
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

    # Check if collection exists and has data
    try:
        collections = client.get_collections().collections
        collection_exists = any(c.name == QDRANT_COLLECTION for c in collections)

        if collection_exists:
            info = client.get_collection(QDRANT_COLLECTION)
            if info.points_count > 0 and not force:
                print(f"  → Qdrant collection '{QDRANT_COLLECTION}' already has {info.points_count} entities")
                print("  → Use --force to rebuild")
                return
            # Delete existing collection
            print(f"  Deleting existing collection '{QDRANT_COLLECTION}'...")
            client.delete_collection(QDRANT_COLLECTION)
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
        metadata = _extract_rich_metadata(info)
        
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

    print(f"  Creating embeddings for {len(entities)} entities...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )

    # Create search texts with rich metadata
    texts: list[str] = []
    for e in entities:
        # Build comprehensive search text
        parts = [f"{e['type']}: {e['name']}"]
        if e["full_name"]:
            parts.append(e["full_name"])
        if e["description"]:
            parts.append(str(e["description"]))
        text = "\n".join(parts)
        texts.append(text)

    # Embed in batches
    batch_size = 1000
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)
        print(f"    Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    # Create collection
    dimension = len(all_vectors[0])
    print(f"  Creating collection with dimension {dimension}...")
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE,
        ),
    )

    # Upload in batches
    print(f"  Uploading {len(entities)} vectors...")
    for i in range(0, len(entities), batch_size):
        batch_entities = entities[i:i + batch_size]
        batch_vectors = all_vectors[i:i + batch_size]

        points = []
        for j, (entity, vector) in enumerate(zip(batch_entities, batch_vectors)):
            points.append(PointStruct(
                id=i + j,
                vector=vector,
                payload={
                    "entity_id": entity["id"],
                    "type": entity["type"],
                    "name": entity["name"],
                    "full_name": entity.get("full_name", ""),
                    "description": entity["description"],
                },
            ))

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"    Uploaded {min(i + batch_size, len(entities))}/{len(entities)}")

    print(f"  ✓ Qdrant index built with {len(entities)} entities")


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

    args = parser.parse_args()

    print("=" * 60)
    print("STaRK-Prime Data Store Builder")
    print("=" * 60)
    print()
    print("Docker services:")
    print(f"  PostgreSQL: {POSTGRES_URL.split('@')[1] if '@' in POSTGRES_URL else POSTGRES_URL}")
    print(f"  Fuseki: {FUSEKI_QUERY_URL}")
    print(f"  Qdrant: {QDRANT_HOST}:{QDRANT_PORT}/{QDRANT_COLLECTION}")
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
            build_qdrant_index(loader, force=args.force)
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
