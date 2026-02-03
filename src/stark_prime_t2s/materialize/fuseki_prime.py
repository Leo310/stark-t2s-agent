"""Apache Jena Fuseki SPARQL endpoint for STaRK-Prime."""

import re
from typing import Any

import requests
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm

from stark_prime_t2s.config import (
    FUSEKI_DATA_URL,
    FUSEKI_QUERY_URL,
    FUSEKI_UPDATE_URL,
    FUSEKI_ADMIN_PASSWORD,
    STARK_PRIME_NS,
)
from stark_prime_t2s.dataset.parse_prime_processed import PrimeDataLoader


def sanitize_uri_part(name: str) -> str:
    """Convert a name to a valid URI component."""
    parts = re.split(r"[^a-zA-Z0-9]+", name)
    return "".join(part.capitalize() for part in parts if part)


class FusekiPrimeStore:
    """Fuseki SPARQL endpoint store for STaRK-Prime knowledge base.

    Uses HTTP requests to communicate with Apache Jena Fuseki server.
    """

    def __init__(
        self,
        query_url: str | None = None,
        update_url: str | None = None,
        data_url: str | None = None,
    ):
        """Initialize the store.

        Args:
            query_url: SPARQL query endpoint URL
            update_url: SPARQL update endpoint URL
            data_url: Graph store protocol URL for bulk data loading
        """
        self.query_url = query_url or FUSEKI_QUERY_URL
        self.update_url = update_url or FUSEKI_UPDATE_URL
        self.data_url = data_url or FUSEKI_DATA_URL
        self.ns = STARK_PRIME_NS

        # Type mappings
        self.node_type_classes: dict[str, str] = {}
        self.edge_type_properties: dict[str, str] = {}

    def execute_sparql(self, query: str) -> list[dict[str, Any]]:
        """Execute a SPARQL query and return results.

        Args:
            query: SPARQL query string

        Returns:
            List of result bindings as dicts

        Raises:
            ValueError: If query is not read-only
        """
        # Validate read-only
        query_upper = query.strip().upper()
        forbidden = ["INSERT", "DELETE", "LOAD", "CLEAR", "DROP", "CREATE", "ADD", "MOVE", "COPY"]
        for keyword in forbidden:
            if keyword in query_upper:
                raise ValueError(f"Query contains forbidden keyword: {keyword}")

        sparql = SPARQLWrapper(self.query_url)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        try:
            results = sparql.query().convert()
        except Exception as e:
            raise ValueError(f"SPARQL query error: {e}")

        # Handle different result types
        if "boolean" in results:
            # ASK query
            return [{"result": results["boolean"]}]

        if "results" not in results or "bindings" not in results["results"]:
            return []

        # SELECT query
        bindings = results["results"]["bindings"]
        return [
            {var: self._convert_sparql_value(binding.get(var)) for var in results["head"]["vars"]}
            for binding in bindings
        ]

    def _convert_sparql_value(self, value: dict | None) -> Any:
        """Convert a SPARQL JSON result value to Python."""
        if value is None:
            return None

        val_type = value.get("type")
        val = value.get("value")

        if val_type == "uri":
            return val
        elif val_type == "literal":
            datatype = value.get("datatype", "")
            if "integer" in datatype:
                return int(val)
            elif "decimal" in datatype or "float" in datatype or "double" in datatype:
                return float(val)
            elif "boolean" in datatype:
                return val.lower() == "true"
            return val
        elif val_type == "bnode":
            return f"_:{val}"

        return val

    def build_from_loader(self, loader: PrimeDataLoader) -> None:
        """Build the RDF graph in Fuseki from a PrimeDataLoader."""
        print(f"Building Fuseki RDF graph...")

        # Clear existing data
        print("  Clearing existing data...")
        self._clear_graph()

        # Build class and property URIs
        print("  Setting up vocabulary...")
        for type_id, type_name in loader.node_type_dict.items():
            class_uri = f"{self.ns}{sanitize_uri_part(type_name)}"
            self.node_type_classes[type_name] = class_uri

        for type_id, type_name in loader.edge_type_dict.items():
            prop_name = sanitize_uri_part(type_name)
            if prop_name:
                prop_name = prop_name[0].lower() + prop_name[1:]
            prop_uri = f"{self.ns}{prop_name}"
            self.edge_type_properties[type_name] = prop_uri

        # Generate and upload triples in batches
        print("  Uploading nodes...")
        batch_size = 5000
        triples_batch = []

        # Add vocabulary definitions
        triples_batch.extend(self._generate_vocab_triples(loader))

        for node in tqdm(loader.iter_nodes(), total=loader.num_nodes, desc="  Nodes"):
            triples_batch.extend(self._node_to_triples(node))

            if len(triples_batch) >= batch_size:
                self._upload_triples(triples_batch)
                triples_batch = []

        if triples_batch:
            self._upload_triples(triples_batch)

        # Upload edges
        print("  Uploading edges...")
        triples_batch = []

        for edge in tqdm(loader.iter_edges(), total=loader.num_edges, desc="  Edges"):
            triples_batch.append(self._edge_to_triple(edge))

            if len(triples_batch) >= batch_size:
                self._upload_triples(triples_batch)
                triples_batch = []

        if triples_batch:
            self._upload_triples(triples_batch)

        print(f"  ✓ Fuseki RDF graph built successfully")

    def _generate_vocab_triples(self, loader: PrimeDataLoader) -> list[str]:
        """Generate vocabulary definition triples in N-Triples format.

        Note: N-Triples does NOT support 'a' shorthand - must use full rdf:type IRI.
        """
        triples = []
        rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        rdfs_class = "http://www.w3.org/2000/01/rdf-schema#Class"
        rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"
        rdf_property = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"

        # Class definitions
        for type_name, class_uri in self.node_type_classes.items():
            triples.append(f"<{class_uri}> <{rdf_type}> <{rdfs_class}> .")
            triples.append(f'<{class_uri}> <{rdfs_label}> "{type_name}" .')

        # Property definitions
        for type_name, prop_uri in self.edge_type_properties.items():
            triples.append(f"<{prop_uri}> <{rdf_type}> <{rdf_property}> .")
            triples.append(f'<{prop_uri}> <{rdfs_label}> "{type_name}" .')

        return triples

    def _node_to_triples(self, node) -> list[str]:
        """Convert a node to N-Triples format.

        Note: N-Triples does NOT support 'a' shorthand - must use full rdf:type IRI.
        """
        triples = []
        node_uri = f"{self.ns}node/{node.node_id}"
        class_uri = self.node_type_classes[node.node_type]
        rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

        # Type (using full IRI, not 'a' shorthand)
        triples.append(f"<{node_uri}> <{rdf_type}> <{class_uri}> .")

        # Node ID
        triples.append(
            f'<{node_uri}> <{self.ns}nodeId> "{node.node_id}"^^<http://www.w3.org/2001/XMLSchema#integer> .'
        )

        # Name
        if node.name:
            escaped_name = self._escape_literal(node.name)
            triples.append(f'<{node_uri}> <{self.ns}name> "{escaped_name}" .')

        # Description
        if node.description:
            escaped_desc = self._escape_literal(
                node.description[:1000]
            )  # Truncate long descriptions
            triples.append(f'<{node_uri}> <{self.ns}description> "{escaped_desc}" .')

        return triples

    def _edge_to_triple(self, edge) -> str:
        """Convert an edge to N-Triple format."""
        src_uri = f"{self.ns}node/{edge.src_id}"
        dst_uri = f"{self.ns}node/{edge.dst_id}"
        prop_uri = self.edge_type_properties[edge.edge_type]

        return f"<{src_uri}> <{prop_uri}> <{dst_uri}> ."

    def _escape_literal(self, text: str) -> str:
        """Escape a string for use in N-Triples literal."""
        return (
            text.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )

    def _clear_graph(self) -> None:
        """Clear all data from the graph using HTTP request with auth."""
        try:
            # Use requests directly with authentication
            response = requests.post(
                self.update_url,
                data="CLEAR ALL",
                headers={"Content-Type": "application/sparql-update"},
                auth=("admin", FUSEKI_ADMIN_PASSWORD),
            )
            if response.status_code not in (200, 204):
                print(
                    f"  Warning: Could not clear graph: {response.status_code} {response.text[:100]}"
                )
        except Exception as e:
            print(f"  Warning: Could not clear graph: {e}")

    def _upload_triples(self, triples: list[str]) -> None:
        """Upload triples to Fuseki using Graph Store Protocol."""
        if not triples:
            return

        nt_data = "\n".join(triples)

        response = requests.post(
            self.data_url,
            data=nt_data.encode("utf-8"),
            headers={"Content-Type": "application/n-triples"},
            auth=("admin", FUSEKI_ADMIN_PASSWORD),
        )

        if response.status_code not in (200, 201, 204):
            raise ValueError(
                f"Failed to upload triples: {response.status_code} {response.text[:200]}"
            )

    def get_vocabulary_summary(self) -> str:
        """Get a summary of the RDF vocabulary for the LLM prompt."""
        lines = [
            "RDF Vocabulary for STaRK-Prime (Fuseki SPARQL Endpoint):",
            f"Namespace: {self.ns} (prefix: sp)",
            f"SPARQL Endpoint: {self.query_url}",
            "",
            "Node URI pattern: sp:node/<id>",
            "",
            "Classes (node types):",
        ]

        # Query for class counts
        try:
            for type_name in sorted(self.node_type_classes.keys()):
                class_uri = self.node_type_classes[type_name]
                result = self.execute_sparql(
                    f"SELECT (COUNT(?s) as ?count) WHERE {{ ?s a <{class_uri}> }}"
                )
                count = result[0]["count"] if result else 0
                lines.append(f"  - sp:{sanitize_uri_part(type_name)} ({count:,} instances)")
        except Exception:
            lines.append("  [Could not fetch class counts]")

        lines.append("")
        lines.append("Properties (edge types):")

        try:
            for type_name in sorted(self.edge_type_properties.keys()):
                prop_uri = self.edge_type_properties[type_name]
                result = self.execute_sparql(
                    f"SELECT (COUNT(*) as ?count) WHERE {{ ?s <{prop_uri}> ?o }}"
                )
                count = result[0]["count"] if result else 0
                prop_name = sanitize_uri_part(type_name)
                if prop_name:
                    prop_name = prop_name[0].lower() + prop_name[1:]
                lines.append(f"  - sp:{prop_name} ({count:,} triples)")
        except Exception:
            lines.append("  [Could not fetch property counts]")

        lines.append("")
        lines.append("Node literal properties:")
        lines.append("  - sp:nodeId (xsd:integer) - the node's unique ID")
        lines.append("  - sp:name (xsd:string) - the node's name")
        lines.append("  - sp:description (xsd:string) - the node's description")

        return "\n".join(lines)

    def load_type_mappings(self) -> None:
        """Load type mappings from the existing graph."""
        # Query for classes
        results = self.execute_sparql(f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?class ?label WHERE {{
                ?class a rdfs:Class .
                ?class rdfs:label ?label .
                FILTER(STRSTARTS(STR(?class), "{self.ns}"))
            }}
        """)
        for row in results:
            self.node_type_classes[row["label"]] = row["class"]

        # Query for properties
        results = self.execute_sparql(f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?prop ?label WHERE {{
                ?prop a rdf:Property .
                ?prop rdfs:label ?label .
                FILTER(STRSTARTS(STR(?prop), "{self.ns}"))
            }}
        """)
        for row in results:
            if row["label"] not in ["name", "description", "nodeId", "rawJson"]:
                self.edge_type_properties[row["label"]] = row["prop"]

    def is_available(self) -> bool:
        """Check if Fuseki is available and has data."""
        try:
            result = self.execute_sparql("SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }")
            return result[0]["count"] > 0
        except Exception:
            return False
