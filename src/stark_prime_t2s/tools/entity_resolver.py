"""Entity resolution using vector similarity search for STaRK-Prime.

This module provides semantic search over entity names and descriptions
to help the agent resolve natural language references to actual entity IDs.

Uses Qdrant vector database (Docker) for persistent storage.
"""

from typing import Any

from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from stark_prime_t2s.config import (
    OPENAI_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_COLLECTION_FULL,
    QDRANT_HOST,
    QDRANT_PORT,
)


class EntityInfo(BaseModel):
    """Information about an entity in the knowledge base."""

    id: int = Field(description="The unique node ID")
    type: str = Field(description="The entity type (e.g., 'disease', 'drug')")
    name: str = Field(description="The entity name")
    description: str | None = Field(default=None, description="Entity description if available")

    def to_search_text(self) -> str:
        """Create text for embedding/search."""
        parts = [f"{self.type}: {self.name}"]
        if self.description:
            desc = self.description[:500] if len(self.description) > 500 else self.description
            parts.append(desc)
        return "\n".join(parts)


class EntitySearchResult(BaseModel):
    """Result from entity search."""

    entities: list[dict[str, Any]] = Field(description="List of matching entities")
    query: str = Field(description="The search query")

    def to_string(self) -> str:
        """Format results for the LLM."""
        if not self.entities:
            return f"No entities found matching '{self.query}'"

        lines = [f"Found {len(self.entities)} entities matching '{self.query}':", ""]

        for i, entity in enumerate(self.entities, 1):
            lines.append(f"{i}. [{entity['type']}] {entity['name']} (ID: {entity['id']})")
            if entity.get("description"):
                lines.append(f"   {entity.get('description', '')}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Qdrant Client
# ---------------------------------------------------------------------------

_qdrant_client: QdrantClient | None = None
_embeddings: OpenAIEmbeddings | None = None


def _get_qdrant_client() -> QdrantClient:
    """Get Qdrant client (singleton)."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client


def _get_embeddings() -> OpenAIEmbeddings:
    """Get OpenAI embeddings instance (singleton)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY,
        )
    return _embeddings


def _collection_exists() -> bool:
    """Check if Qdrant collection exists and has data."""
    try:
        client = _get_qdrant_client()
        collections = client.get_collections().collections
        for c in collections:
            if c.name == QDRANT_COLLECTION:
                info = client.get_collection(QDRANT_COLLECTION)
                return info.points_count > 0
        return False
    except Exception:
        return False


def _normalize_entity_type(entity_type: str) -> str:
    """Normalize user-provided entity_type to the canonical type used in the KB."""
    et = entity_type.strip().lower()

    # The Prime KB uses slashes for a couple of types; keep tool UX underscore-friendly.
    slash_mappings = {
        "gene_protein": "gene/protein",
        "gene/protein": "gene/protein",
        "effect_phenotype": "effect/phenotype",
        "effect/phenotype": "effect/phenotype",
    }
    if et in slash_mappings:
        return slash_mappings[et]

    return et


def _get_collection_name(use_full: bool) -> str:
    return QDRANT_COLLECTION_FULL if use_full else QDRANT_COLLECTION


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_entity_index(_force_rebuild: bool = False) -> int:
    """Check if the Qdrant entity index is ready.

    Args:
        _force_rebuild: Ignored (index is managed in Qdrant)

    Returns:
        Number of entities in the index
    """
    if _collection_exists():
        client = _get_qdrant_client()
        info = client.get_collection(QDRANT_COLLECTION)
        print(f"  ✓ Qdrant collection '{QDRANT_COLLECTION}' ready ({info.points_count} entities)")
        return info.points_count
    else:
        print(f"  Warning: Qdrant collection '{QDRANT_COLLECTION}' not found or empty")
        print("  Run: python scripts/build_prime_stores.py")
        return 0


def search_entities(
    query: str,
    entity_type: str | None = None,
    top_k: int = 5,
    use_full: bool = True,
) -> EntitySearchResult:
    """Search for entities matching a query.

    Args:
        query: The search query (natural language)
        entity_type: Optional filter by entity type
        top_k: Maximum number of results to return

    Returns:
        EntitySearchResult with matching entities
    """
    try:
        client = _get_qdrant_client()
        embeddings = _get_embeddings()

        # Embed the query
        query_vector = embeddings.embed_query(query)

        # Build filter if type specified
        search_filter = None
        if entity_type:
            canonical_type = _normalize_entity_type(entity_type)
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=canonical_type),
                    )
                ]
            )

        # Search
        results = client.query_points(
            collection_name=_get_collection_name(use_full),
            query=query_vector,
            limit=top_k,
            query_filter=search_filter,
        )

        # Convert to entity dicts
        entities = []
        for hit in results.points:
            name = hit.payload["name"]
            full_name = hit.payload.get("full_name", "")
            # Show full name in parentheses if different from short name
            display_name = (
                f"{name} ({full_name})" if full_name and full_name.lower() != name.lower() else name
            )

            entities.append(
                {
                    "id": hit.payload["entity_id"],
                    "type": hit.payload["type"],
                    "name": display_name,
                    "description": hit.payload.get("description", ""),
                }
            )

        return EntitySearchResult(entities=entities, query=query)

    except Exception as e:
        print(f"Qdrant search error: {e}")
        return EntitySearchResult(entities=[], query=query)


# ---------------------------------------------------------------------------
# LangChain Tool
# ---------------------------------------------------------------------------


@tool
def search_entities_tool(
    query: str,
    entity_type: str | None = None,
    use_full: bool = True,
) -> str:
    """Search for entities in the STaRK-Prime knowledge base by name or description.

    Use this tool FIRST to find the correct entity IDs before writing SQL/SPARQL queries.
    This performs semantic search, so it can find entities even with synonyms or partial matches.

    Args:
        query: Natural language description of the entity you're looking for.
               Examples: "Alzheimer's disease", "aspirin", "insulin receptor", "cell death pathway"
        entity_type: Optional filter by entity type. Valid types include:
               - "disease" - Medical conditions and diseases
               - "drug" - Pharmaceutical compounds and medications
               - "gene_protein" (or "gene/protein") - Genes and proteins
               - "effect_phenotype" (or "effect/phenotype") - Effects and phenotypes
               - "pathway" - Biological pathways
               - "anatomy" - Anatomical structures
               - "molecular_function" - Molecular functions
               - "cellular_component" - Cellular components
               - "biological_process" - Biological processes
               - "exposure" - Environmental exposures
               Leave empty to search all entity types.
        use_full: Use the full-description Qdrant index (default True). Set False
                  to search the shorter/truncated index.

    Returns:
        A list of matching entities with their IDs, types, names, and descriptions.
        Use the returned IDs in your SQL/SPARQL queries.

    Example:
        search_entities_tool("breast cancer", "disease")
        → Returns diseases related to breast cancer with their IDs

        Then use the ID in SQL:
        SELECT dst_id FROM indication WHERE src_id = <returned_id>
    """
    result = search_entities(
        query,
        entity_type=entity_type,
        top_k=5,
        use_full=use_full,
    )
    return result.to_string()


def get_search_entities_tool():
    """Get the search_entities tool for use with create_agent."""
    return search_entities_tool


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Checking entity index...")
    count = build_entity_index()
    print(f"\nIndexed {count} entities")

    if count > 0:
        print("\n" + "=" * 60)
        print("Testing entity search...")
        print("=" * 60)

        test_queries = [
            ("Alzheimer's disease", "disease"),
            ("aspirin", "drug"),
            ("insulin", None),
            ("cell death", "biological_process"),
        ]

        for q, etype in test_queries:
            print(f"\nQuery: '{q}'" + (f" (type: {etype})" if etype else ""))
            print("-" * 40)
            res = search_entities(q, entity_type=etype, top_k=5)
            print(res.to_string())
