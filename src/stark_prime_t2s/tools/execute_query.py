"""SQL/SPARQL execution tool for STaRK-Prime agent."""

import re
from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, Field

from stark_prime_t2s.config import MAX_QUERY_ROWS


class QueryResult(BaseModel):
    """Result from executing a query."""

    success: bool = Field(description="Whether the query executed successfully")
    language: str = Field(description="The query language used (sql or sparql)")
    columns: list[str] = Field(default_factory=list, description="Column names")
    rows: list[dict[str, Any]] = Field(default_factory=list, description="Result rows")
    row_count: int = Field(default=0, description="Number of rows returned")
    truncated: bool = Field(default=False, description="Whether results were truncated")
    error: str | None = Field(default=None, description="Error message if failed")

    def to_string(self) -> str:
        """Convert result to a string for the LLM."""
        if not self.success:
            return f"Error executing {self.language} query: {self.error}"

        if not self.rows:
            return "Query returned 0 rows."

        lines = [
            f"Query returned {self.row_count} row(s){' (truncated)' if self.truncated else ''}:"
        ]
        lines.append("")

        if self.columns:
            lines.append(" | ".join(self.columns))
            lines.append("-" * len(lines[-1]))

        for row in self.rows:
            values = (
                [str(row.get(col, "")) for col in self.columns]
                if self.columns
                else [str(v) for v in row.values()]
            )
            lines.append(" | ".join(values))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Store instances (lazily initialized)
# ---------------------------------------------------------------------------

_sql_store = None
_sparql_store = None


def get_sql_store():
    """Get the PostgreSQL store instance."""
    global _sql_store
    if _sql_store is None:
        from stark_prime_t2s.materialize.postgres_prime import PostgresPrimeStore

        _sql_store = PostgresPrimeStore()
        _sql_store.load_table_mappings()
        # Ensure unified views exist and are up-to-date (normalized types/edge_type)
        _sql_store._create_unified_views()
    return _sql_store


def get_sparql_store():
    """Get the Fuseki SPARQL store instance."""
    global _sparql_store
    if _sparql_store is None:
        from stark_prime_t2s.materialize.fuseki_prime import FusekiPrimeStore

        _sparql_store = FusekiPrimeStore()
        _sparql_store.load_type_mappings()
    return _sparql_store


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


def execute_sql_query(query: str) -> QueryResult:
    """Execute a SQL query against PostgreSQL."""
    try:
        store = get_sql_store()
        rows = store.execute_read_only(query)

        truncated = len(rows) > MAX_QUERY_ROWS
        if truncated:
            rows = rows[:MAX_QUERY_ROWS]

        columns = list(rows[0].keys()) if rows else []

        return QueryResult(
            success=True,
            language="sql",
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )
    except Exception as e:
        return QueryResult(
            success=False,
            language="sql",
            error=str(e),
        )


def _normalize_sparql_query(query: str) -> str:
    """Normalize common SPARQL prefix/IRI mistakes."""
    cleaned = query.strip()
    if "PREFIX sp:" in cleaned and "<http://stark.stanford.edu/prime/>" not in cleaned:
        cleaned = cleaned.replace(
            "PREFIX sp: http://stark.stanford.edu/prime/",
            "PREFIX sp: <http://stark.stanford.edu/prime/>",
        )

    cleaned = re.sub(
        r"<http://stark\.stanford\.edu/prime/node/>\s*([A-Za-z0-9_-]+)",
        r"<http://stark.stanford.edu/prime/node/\1>",
        cleaned,
    )
    cleaned = re.sub(
        r"\bsp:node/([A-Za-z0-9_-]+)\b",
        r"<http://stark.stanford.edu/prime/node/\1>",
        cleaned,
    )
    return cleaned


def execute_sparql_query(query: str) -> QueryResult:
    """Execute a SPARQL query against Fuseki."""
    try:
        store = get_sparql_store()
        normalized = _normalize_sparql_query(query)
        rows = store.execute_sparql(normalized)

        truncated = len(rows) > MAX_QUERY_ROWS
        if truncated:
            rows = rows[:MAX_QUERY_ROWS]

        columns = list(rows[0].keys()) if rows else []

        return QueryResult(
            success=True,
            language="sparql",
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )
    except Exception as e:
        return QueryResult(
            success=False,
            language="sparql",
            error=str(e),
        )


def execute_query(language: str, query: str) -> str:
    """Execute a SQL or SPARQL query."""
    language_lower = language.lower().strip()

    if language_lower == "sql":
        result = execute_sql_query(query)
    elif language_lower == "sparql":
        result = execute_sparql_query(query)
    else:
        return f"Error: Unknown query language '{language}'. Use 'sql' or 'sparql'."

    return result.to_string()


@tool
def execute_query_tool(language: str, query: str) -> str:
    """Execute a SQL or SPARQL query against the STaRK-Prime biomedical knowledge base.

    Use this tool to query the knowledge base for information about diseases, drugs,
    genes/proteins, pathways, molecular functions, and their relationships.

    Args:
        language: The query language to use. Must be either:
            - "sql" for SQL queries against PostgreSQL
            - "sparql" for SPARQL queries against Fuseki
        query: The query string to execute. Must be read-only:
            - SQL: Only SELECT statements allowed
            - SPARQL: Only SELECT, ASK, CONSTRUCT, DESCRIBE allowed

    Returns:
        The query results formatted as a table, or an error message if the query failed.

    Examples:
        SQL: SELECT * FROM disease LIMIT 5
        SPARQL: SELECT ?node ?name WHERE { ?node a sp:Disease . ?node sp:name ?name } LIMIT 5
    """
    return execute_query(language, query)


@tool
def execute_sql_query_tool(query: str) -> str:
    """Execute a SQL query against the STaRK-Prime biomedical knowledge base.

    Args:
        query: The SQL query string to execute. Must be read-only (SELECT only).

    Returns:
        The query results formatted as a table, or an error message if the query failed.
    """
    return execute_query("sql", query)


@tool
def execute_sparql_query_tool(query: str) -> str:
    """Execute a SPARQL query against the STaRK-Prime biomedical knowledge base.

    Args:
        query: The SPARQL query string to execute. Must be read-only.

    Returns:
        The query results formatted as a table, or an error message if the query failed.
    """
    return execute_query("sparql", query)


def get_execute_query_tool():
    """Get the execute_query tool for use with create_agent."""
    return execute_query_tool


def get_execute_sql_query_tool():
    """Get the SQL-only execute_query tool for use with create_agent."""
    return execute_sql_query_tool


def get_execute_sparql_query_tool():
    """Get the SPARQL-only execute_query tool for use with create_agent."""
    return execute_sparql_query_tool


def get_sql_schema_summary() -> str:
    """Get a summary of the SQL schema only."""
    parts = []

    try:
        store = get_sql_store()
        if store.is_available():
            parts.append("=" * 60)
            parts.append("SQL DATABASE (PostgreSQL)")
            parts.append("=" * 60)
            parts.append(store.get_schema_summary())
            try:
                edge_rows = store.execute_read_only(
                    """
                    SELECT edge_type, src_type, dst_type, COUNT(*) AS cnt
                    FROM all_edges
                    GROUP BY edge_type, src_type, dst_type
                    ORDER BY cnt DESC;
                    """
                )
                if edge_rows:
                    parts.append("")
                    parts.append("EDGE TYPE SUMMARY (from all_edges):")
                    for row in edge_rows:
                        edge_type = row.get("edge_type")
                        src_type = row.get("src_type")
                        dst_type = row.get("dst_type")
                        cnt = row.get("cnt")
                        parts.append(f"  {edge_type}: {src_type} -> {dst_type} ({cnt:,} rows)")
            except Exception as e:
                parts.append("")
                parts.append(f"[Edge summary unavailable: {e}]")
        else:
            parts.append("[PostgreSQL database not available or empty]")
    except Exception as e:
        parts.append(f"[PostgreSQL unavailable: {e}]")

    return "\n".join(parts)


def get_sparql_vocab_summary() -> str:
    """Get a summary of the SPARQL vocabulary only."""
    parts = []

    try:
        store = get_sparql_store()
        if store.is_available():
            parts.append("=" * 60)
            parts.append("RDF GRAPH (Fuseki SPARQL)")
            parts.append("=" * 60)
            parts.append(store.get_vocabulary_summary())
        else:
            parts.append("[Fuseki SPARQL endpoint not available or empty]")
    except Exception as e:
        parts.append(f"[Fuseki unavailable: {e}]")

    return "\n".join(parts)


def get_schema_and_vocab_summary() -> str:
    """Get a combined summary of the SQL schema and RDF vocabulary."""
    parts = [get_sql_schema_summary(), "", get_sparql_vocab_summary()]
    return "\n".join(parts)


if __name__ == "__main__":
    print("Testing execute_query tool...")
    print()

    print("SQL Test:")
    result = execute_query("sql", "SELECT id, name FROM disease LIMIT 5")
    print(result)
    print()

    print("SPARQL Test:")
    result = execute_query(
        "sparql",
        """
        PREFIX sp: <http://stark.stanford.edu/prime/>
        SELECT ?node ?name WHERE {
            ?node sp:name ?name
        } LIMIT 5
    """,
    )
    print(result)
