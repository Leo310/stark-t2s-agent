"""PostgreSQL materialization for STaRK-Prime with typed schema.

Creates one table per entity type (disease, drug, gene_protein, etc.)
and one table per relation type (associated_with, side_effect, etc.)
following the STaRK-Prime schema.
"""

import re
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from tqdm import tqdm

from stark_prime_t2s.config import POSTGRES_URL
from stark_prime_t2s.dataset.parse_prime_processed import PrimeDataLoader


def sanitize_table_name(name: str) -> str:
    """Convert a type name to a valid PostgreSQL table name."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "t_" + name
    return name.lower()


class PostgresPrimeStore:
    """PostgreSQL store for STaRK-Prime knowledge base with typed schema.

    Creates typed tables directly named after entity and relation types:
    - Entity tables: disease, drug, gene_protein, pathway, etc.
    - Relation tables: associated_with, indication, side_effect, etc.

    Also creates unified views (all_nodes, all_edges) for cross-type queries.
    """

    def __init__(self, db_url: str | None = None):
        """Initialize the store."""
        self.db_url = db_url or POSTGRES_URL
        self._engine = None

        # Table name mappings (type_name -> table_name)
        self.node_type_tables: dict[str, str] = {}
        self.edge_type_tables: dict[str, str] = {}

    @property
    def engine(self):
        """Get the SQLAlchemy engine (lazy initialization with connection pooling)."""
        if self._engine is None:
            self._engine = create_engine(
                self.db_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
        return self._engine

    def execute(self, query: str, params: dict | None = None) -> list[dict[str, Any]]:
        """Execute a SQL query and return results."""
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            if result.returns_rows:
                columns = list(result.keys())
                return [dict(zip(columns, row)) for row in result.fetchall()]
            return []

    def execute_read_only(self, query: str) -> list[dict[str, Any]]:
        """Execute a read-only SQL query."""
        import re

        # Strip SQL comments before validation
        # Remove -- line comments
        query_no_comments = re.sub(r"--[^\n]*", "", query)
        # Remove /* block comments */
        query_no_comments = re.sub(r"/\*.*?\*/", "", query_no_comments, flags=re.DOTALL)

        query_upper = query_no_comments.strip().upper()
        if not (query_upper.startswith("SELECT") or query_upper.startswith("WITH")):
            raise ValueError("Only SELECT queries are allowed")

        dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "REPLACE"]
        for keyword in dangerous:
            # Check for dangerous keywords as whole words (not inside identifiers)
            if re.search(rf"\b{keyword}\b", query_upper):
                raise ValueError(f"Query contains forbidden keyword: {keyword}")

        return self.execute(query)

    def build_from_loader(self, loader: PrimeDataLoader) -> None:
        """Build the PostgreSQL database from a PrimeDataLoader.

        Creates typed tables following the plan:
        - One table per entity type (e.g., disease, drug, gene_protein)
        - One table per relation type (e.g., associated_with, side_effect)
        - Unified views for cross-type queries (all_nodes, all_edges)
        """
        print(f"Building PostgreSQL database...")

        with self.engine.connect() as conn:
            # Drop existing views first
            print("  Dropping existing views...")
            conn.execute(text("DROP VIEW IF EXISTS all_edges CASCADE"))
            conn.execute(text("DROP VIEW IF EXISTS all_nodes CASCADE"))
            conn.execute(text("DROP VIEW IF EXISTS all_edges_raw CASCADE"))
            conn.execute(text("DROP VIEW IF EXISTS all_nodes_raw CASCADE"))

            # Drop existing tables
            print("  Dropping existing tables...")
            for type_name in loader.node_type_dict.values():
                table_name = sanitize_table_name(type_name)
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))

            for type_name in loader.edge_type_dict.values():
                table_name = sanitize_table_name(type_name)
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))

            conn.commit()

            # Create entity type tables (typed schema)
            print("  Creating entity type tables...")
            for type_id, type_name in loader.node_type_dict.items():
                table_name = sanitize_table_name(type_name)
                self.node_type_tables[type_name] = table_name

                conn.execute(
                    text(f"""
                    CREATE TABLE {table_name} (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        details TEXT,
                        raw_json TEXT
                    )
                """)
                )
                print(f"    → {table_name}")

            # Create relation type tables (typed schema)
            print("  Creating relation type tables...")
            for type_id, type_name in loader.edge_type_dict.items():
                table_name = sanitize_table_name(type_name)
                self.edge_type_tables[type_name] = table_name

                conn.execute(
                    text(f"""
                    CREATE TABLE {table_name} (
                        src_id INTEGER,
                        dst_id INTEGER,
                        src_type TEXT,
                        dst_type TEXT,
                        PRIMARY KEY (src_id, dst_id)
                    )
                """)
                )
                print(f"    → {table_name}")

            conn.commit()

        # Insert nodes into typed tables
        print("  Inserting nodes...")
        node_batches: dict[str, list] = {t: [] for t in self.node_type_tables}

        for node in tqdm(loader.iter_nodes(), total=loader.num_nodes, desc="  Nodes"):
            row = {
                "id": node.node_id,
                "name": node.name,
                "details": node.to_details_json(),
                "raw_json": node.to_json(),
            }
            node_batches[node.node_type].append(row)

            # Batch insert every 5000 rows per type
            if len(node_batches[node.node_type]) >= 5000:
                self._insert_node_batch(node.node_type, node_batches[node.node_type])
                node_batches[node.node_type] = []

        # Flush remaining nodes
        for type_name, rows in node_batches.items():
            if rows:
                self._insert_node_batch(type_name, rows)

        # Insert edges into typed tables
        print("  Inserting edges...")
        edge_batches: dict[str, list] = {t: [] for t in self.edge_type_tables}

        for edge in tqdm(loader.iter_edges(), total=loader.num_edges, desc="  Edges"):
            row = {
                "src_id": edge.src_id,
                "dst_id": edge.dst_id,
                "src_type": edge.src_type,
                "dst_type": edge.dst_type,
            }
            edge_batches[edge.edge_type].append(row)

            # Batch insert every 10000 rows per type
            if len(edge_batches[edge.edge_type]) >= 10000:
                self._insert_edge_batch(edge.edge_type, edge_batches[edge.edge_type])
                edge_batches[edge.edge_type] = []

        # Flush remaining edges
        for type_name, rows in edge_batches.items():
            if rows:
                self._insert_edge_batch(type_name, rows)

        # Create unified views
        print("  Creating unified views...")
        self._create_unified_views()

        # Create indexes
        print("  Creating indexes...")
        self._create_indexes()

        print(f"  ✓ PostgreSQL database built successfully")

    def _insert_node_batch(self, type_name: str, rows: list) -> None:
        """Insert a batch of nodes into a typed table."""
        if not rows:
            return
        table_name = self.node_type_tables[type_name]
        with self.engine.connect() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {table_name} (id, name, details, raw_json) "
                    f"VALUES (:id, :name, :details, :raw_json) "
                    f"ON CONFLICT (id) DO NOTHING"
                ),
                rows,
            )
            conn.commit()

    def _insert_edge_batch(self, type_name: str, rows: list) -> None:
        """Insert a batch of edges into a typed table."""
        if not rows:
            return
        table_name = self.edge_type_tables[type_name]
        with self.engine.connect() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {table_name} (src_id, dst_id, src_type, dst_type) "
                    f"VALUES (:src_id, :dst_id, :src_type, :dst_type) "
                    f"ON CONFLICT DO NOTHING"
                ),
                rows,
            )
            conn.commit()

    def _create_unified_views(self) -> None:
        """Create unified views for cross-type queries."""
        with self.engine.connect() as conn:
            # Use advisory lock to serialize view creation across processes
            # Lock ID 12345 is arbitrary but unique for this operation
            lock_acquired = conn.execute(text("SELECT pg_try_advisory_lock(12345)")).scalar()

            if not lock_acquired:
                # Another process is creating views, wait a bit and check if they exist
                import time

                time.sleep(1)

            # Check if views already exist to avoid unnecessary DDL in multi-process scenarios
            existing_views = {
                row["table_name"]
                for row in conn.execute(
                    text("""
                        SELECT table_name FROM information_schema.views 
                        WHERE table_schema = 'public' 
                        AND table_name IN ('all_nodes', 'all_nodes_raw', 'all_edges', 'all_edges_raw')
                    """)
                ).mappings()
            }
            if existing_views == {"all_nodes", "all_nodes_raw", "all_edges", "all_edges_raw"}:
                # All views exist, skip creation
                if lock_acquired:
                    conn.execute(text("SELECT pg_advisory_unlock(12345)"))
                # Views already exist - no DDL needed
                return

            # Log that we're creating views (this should happen rarely)
            import os

            print(
                f"  [postgres_prime] Creating unified views (pid={os.getpid()}, lock={lock_acquired})"
            )

            def _normalize_sql(col_expr: str) -> str:
                """SQL expression to normalize a label like sanitize_table_name().

                - Replace non [a-z0-9_] with underscore
                - Collapse multiple underscores
                - Trim leading/trailing underscores
                - Lowercase
                """
                return (
                    "regexp_replace("
                    "regexp_replace("
                    "regexp_replace(lower(coalesce(" + col_expr + ", '')), '[^a-z0-9_]', '_', 'g'),"
                    " '_+', '_', 'g'),"
                    " '^_+|_+$', '', 'g')"
                )

            # Build UNION ALL for all node tables
            node_unions = []
            for type_name, table_name in self.node_type_tables.items():
                node_unions.append(
                    f"SELECT id, '{sanitize_table_name(type_name)}' as type, name, details, raw_json FROM {table_name}"
                )

            if node_unions:
                conn.execute(
                    text(f"""
                    CREATE OR REPLACE VIEW all_nodes AS
                    {" UNION ALL ".join(node_unions)}
                """)
                )
                # Raw view (debug/backcompat)
                raw_node_unions = []
                for type_name, table_name in self.node_type_tables.items():
                    raw_node_unions.append(
                        f"SELECT id, '{type_name}' as type, name, details, raw_json FROM {table_name}"
                    )
                conn.execute(
                    text(f"""
                    CREATE OR REPLACE VIEW all_nodes_raw AS
                    {" UNION ALL ".join(raw_node_unions)}
                """)
                )

            # Build UNION ALL for all edge tables
            edge_unions = []
            for type_name, table_name in self.edge_type_tables.items():
                edge_unions.append(
                    "SELECT "
                    "src_id, "
                    "dst_id, "
                    f"'{sanitize_table_name(type_name)}' as edge_type, "
                    f"{_normalize_sql('src_type')} as src_type, "
                    f"{_normalize_sql('dst_type')} as dst_type "
                    f"FROM {table_name}"
                )

            if edge_unions:
                conn.execute(
                    text(f"""
                    CREATE OR REPLACE VIEW all_edges AS
                    {" UNION ALL ".join(edge_unions)}
                """)
                )
                # Raw view (debug/backcompat)
                raw_edge_unions = []
                for type_name, table_name in self.edge_type_tables.items():
                    raw_edge_unions.append(
                        f"SELECT src_id, dst_id, '{type_name}' as edge_type, src_type, dst_type FROM {table_name}"
                    )
                conn.execute(
                    text(f"""
                    CREATE OR REPLACE VIEW all_edges_raw AS
                    {" UNION ALL ".join(raw_edge_unions)}
                """)
                )

            conn.commit()

            # Release advisory lock if we acquired it
            if lock_acquired:
                conn.execute(text("SELECT pg_advisory_unlock(12345)"))

    def _create_indexes(self) -> None:
        """Create indexes for efficient querying."""
        with self.engine.connect() as conn:
            # Indexes on typed node tables
            for table_name in self.node_type_tables.values():
                conn.execute(
                    text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_name ON {table_name}(name)")
                )

            # Indexes on typed edge tables
            for table_name in self.edge_type_tables.values():
                conn.execute(
                    text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_src ON {table_name}(src_id)")
                )
                conn.execute(
                    text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_dst ON {table_name}(dst_id)")
                )

            conn.commit()

    def get_schema_summary(self) -> str:
        """Get a summary of the database schema for the LLM prompt."""
        lines = ["PostgreSQL Database Schema for STaRK-Prime (Typed Tables):", ""]

        # Entity tables
        lines.append("ENTITY TABLES (one per type):")
        for type_name, table_name in sorted(self.node_type_tables.items()):
            result = self.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
            count = result[0]["cnt"] if result else 0
            lines.append(f"  {table_name} ({count:,} rows) - {type_name}")
            lines.append(
                f"    Columns: id (INTEGER PK), name (TEXT), details (JSON), raw_json (JSON)"
            )

        lines.append("")
        lines.append("RELATION TABLES (one per type):")
        for type_name, table_name in sorted(self.edge_type_tables.items()):
            result = self.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
            count = result[0]["cnt"] if result else 0
            lines.append(f"  {table_name} ({count:,} rows) - {type_name}")
            lines.append(
                f"    Columns: src_id (INTEGER), dst_id (INTEGER), src_type (TEXT), dst_type (TEXT)"
            )

        lines.append("")
        lines.append("UNIFIED VIEWS (for cross-type queries):")
        lines.append("  all_nodes - view combining all entity tables")
        lines.append("    Columns: id, type, name, details, raw_json")
        lines.append("  all_edges - view combining all relation tables")
        lines.append("    Columns: src_id, dst_id, edge_type, src_type, dst_type")

        lines.append("")
        lines.append("COLUMN USAGE NOTES:")
        lines.append("  - 'name': The entity's display name (always populated)")
        lines.append("  - 'details': JSON object with all non-empty type-specific attributes:")
        lines.append("      * disease: mondo_id, mondo_definition, mondo_name, umls_description,")
        lines.append(
            "                 orphanet_definition, orphanet_prevalence, mayo_symptoms, etc."
        )
        lines.append(
            "      * drug: description, mechanism_of_action, indication, pharmacodynamics,"
        )
        lines.append("              protein_binding, half_life, category, group, etc.")
        lines.append("      * gene/protein: _id, name, summary, alias (list), genomic_pos, etc.")
        lines.append("      * pathway: stId, displayName, summation, literatureReference,")
        lines.append("                 goBiologicalProcess, orthologousEvent, etc.")
        lines.append(
            "      * other types: anatomy, biological_process, etc. have minimal/no details"
        )
        lines.append(
            "  - 'raw_json': Full original entity data as JSON (includes 'details' + metadata)"
        )
        lines.append("")
        lines.append("QUERYING DETAILS:")
        lines.append("  Access specific attributes using JSON operators:")
        lines.append("    - Get value: details::json->>'attribute_name'")
        lines.append("    - Check existence: details::json ? 'attribute_name'")
        lines.append("    - Get nested: details::json->>'nested'->>'key'")
        lines.append("  Examples:")
        lines.append(
            "    SELECT name, details::json->>'mondo_definition' FROM disease WHERE id = 27158"
        )
        lines.append(
            "    SELECT name FROM drug WHERE details::json->>'mechanism_of_action' LIKE '%inhibitor%'"
        )
        lines.append("    SELECT * FROM gene_protein WHERE details::json->>'alias' ? 'TP53'")

        return "\n".join(lines)

    def load_table_mappings(self) -> None:
        """Load table name mappings from existing database."""
        # Get all tables
        tables = self.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
        )

        for row in tables:
            table_name = row["table_name"]

            # Ignore legacy / generic schema tables from earlier iterations.
            # We only want the typed schema tables (e.g., disease, gene_protein, target, interacts_with).
            if table_name in {"nodes", "edges"}:
                continue
            if table_name.startswith("node_") or table_name.startswith("edge_"):
                continue

            # Check columns to determine type
            columns = self.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = :table",
                {"table": table_name},
            )
            col_names = [c["column_name"] for c in columns]

            if "id" in col_names and "raw_json" in col_names:
                # Entity table
                self.node_type_tables[table_name] = table_name
            elif "src_id" in col_names and "dst_id" in col_names:
                # Relation table
                self.edge_type_tables[table_name] = table_name

    def is_available(self) -> bool:
        """Check if PostgreSQL is available and has data."""
        try:
            # Check if we have any entity tables with data
            if not self.node_type_tables:
                self.load_table_mappings()

            if self.node_type_tables:
                # Pick first table and check count
                first_table = next(iter(self.node_type_tables.values()))
                result = self.execute(f"SELECT COUNT(*) as cnt FROM {first_table}")
                return result[0]["cnt"] > 0
            return False
        except Exception:
            return False
