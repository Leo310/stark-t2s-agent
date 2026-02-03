"""Parse STaRK-Prime processed data files."""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from stark_prime_t2s.config import PRIME_PROCESSED_DIR


@dataclass
class NodeData:
    """Data for a single node in the Prime KB."""

    node_id: int
    node_type: str
    node_type_id: int
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str | None:
        """Get the node name if available."""
        return self.info.get("name") or self.info.get("title")

    @property
    def description(self) -> str | None:
        """Get the node description if available.

        Looks for descriptions in the details sub-dictionary first,
        then falls back to top-level description/text fields.
        """
        # First check details for type-specific descriptions
        details = self.info.get("details", {})

        # Try type-specific description fields in order of preference
        if self.node_type == "disease":
            desc = (
                details.get("mondo_definition")
                or details.get("umls_description")
                or details.get("orphanet_definition")
            )
            if desc and str(desc).lower() not in ("nan", "none", ""):
                return str(desc)

        elif self.node_type == "drug":
            desc = (
                details.get("description")
                or details.get("mechanism_of_action")
                or details.get("indication")
            )
            if desc and str(desc).lower() not in ("nan", "none", ""):
                return str(desc)

        elif self.node_type == "gene/protein":
            desc = details.get("summary") or details.get("name")
            if desc and str(desc).lower() not in ("nan", "none", ""):
                return str(desc)

        elif self.node_type == "pathway":
            desc = details.get("summation") or details.get("definition")
            if desc and str(desc).lower() not in ("nan", "none", ""):
                return str(desc)

        # Fallback to top-level fields
        return self.info.get("description") or self.info.get("text")

    @property
    def filtered_details(self) -> dict[str, Any]:
        """Get the details dictionary with only non-NaN/non-empty values.

        Filters out None, empty strings, 'nan', 'none', etc.
        Returns a clean dict suitable for storing in the details column.
        """
        details = self.info.get("details", {})
        filtered = {}

        for key, value in details.items():
            # Skip None
            if value is None:
                continue
            # Skip float NaN
            if isinstance(value, float):
                import math

                if math.isnan(value):
                    continue
            # Skip string placeholders
            if isinstance(value, str):
                cleaned = value.strip().lower()
                if cleaned in ("nan", "none", "", "null", "n/a"):
                    continue
            # Include everything else (including empty lists, False, 0, etc.)
            filtered[key] = value

        return filtered

    def to_json(self) -> str:
        """Serialize info dict to JSON."""
        return json.dumps(self.info, ensure_ascii=False)

    def to_details_json(self) -> str:
        """Serialize filtered details dict to JSON for SQL storage."""
        return json.dumps(self.filtered_details, ensure_ascii=False)


@dataclass
class EdgeData:
    """Data for a single edge in the Prime KB."""

    src_id: int
    dst_id: int
    edge_type: str
    edge_type_id: int
    src_type: str | None = None
    dst_type: str | None = None


class PrimeDataLoader:
    """Loader for STaRK-Prime processed data.

    Loads and provides access to:
    - Node type dictionary (type_id -> type_name)
    - Edge type dictionary (type_id -> type_name)
    - Node info (node_id -> dict of attributes)
    - Node types (node_id -> type_id)
    - Edge index (source/target node pairs)
    - Edge types (edge_id -> type_id)
    """

    def __init__(self, processed_dir: Path | None = None):
        """Initialize the loader.

        Args:
            processed_dir: Path to the processed directory.
                          Defaults to config.PRIME_PROCESSED_DIR.
        """
        self.processed_dir = Path(processed_dir or PRIME_PROCESSED_DIR)

        # Lazy-loaded data
        self._node_type_dict: dict[int, str] | None = None
        self._edge_type_dict: dict[int, str] | None = None
        self._node_info: dict[int, dict[str, Any]] | None = None
        self._node_types: torch.Tensor | None = None
        self._edge_index: torch.Tensor | None = None
        self._edge_types: torch.Tensor | None = None

    def _load_pickle(self, filename: str) -> Any:
        """Load a pickle file from the processed directory."""
        path = self.processed_dir / filename
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_torch(self, filename: str) -> torch.Tensor:
        """Load a torch tensor from the processed directory."""
        path = self.processed_dir / filename
        return torch.load(path, map_location="cpu", weights_only=True)

    @property
    def node_type_dict(self) -> dict[int, str]:
        """Get the node type dictionary (type_id -> type_name)."""
        if self._node_type_dict is None:
            self._node_type_dict = self._load_pickle("node_type_dict.pkl")
        return self._node_type_dict

    @property
    def edge_type_dict(self) -> dict[int, str]:
        """Get the edge type dictionary (type_id -> type_name)."""
        if self._edge_type_dict is None:
            self._edge_type_dict = self._load_pickle("edge_type_dict.pkl")
        return self._edge_type_dict

    @property
    def node_info(self) -> dict[int, dict[str, Any]]:
        """Get the node info dictionary (node_id -> attributes dict)."""
        if self._node_info is None:
            self._node_info = self._load_pickle("node_info.pkl")
        return self._node_info

    @property
    def node_types(self) -> torch.Tensor:
        """Get node types tensor (node_id -> type_id)."""
        if self._node_types is None:
            self._node_types = self._load_torch("node_types.pt")
        return self._node_types

    @property
    def edge_index(self) -> torch.Tensor:
        """Get edge index tensor (2 x num_edges, [src_ids, dst_ids])."""
        if self._edge_index is None:
            self._edge_index = self._load_torch("edge_index.pt")
        return self._edge_index

    @property
    def edge_types(self) -> torch.Tensor:
        """Get edge types tensor (edge_id -> type_id)."""
        if self._edge_types is None:
            self._edge_types = self._load_torch("edge_types.pt")
        return self._edge_types

    @property
    def num_nodes(self) -> int:
        """Get the number of nodes."""
        return len(self.node_types)

    @property
    def num_edges(self) -> int:
        """Get the number of edges."""
        return self.edge_index.shape[1]

    @property
    def node_type_names(self) -> list[str]:
        """Get sorted list of node type names."""
        return sorted(self.node_type_dict.values())

    @property
    def edge_type_names(self) -> list[str]:
        """Get sorted list of edge type names."""
        return sorted(self.edge_type_dict.values())

    def get_node_type_name(self, type_id: int) -> str:
        """Get the name for a node type ID."""
        return self.node_type_dict[type_id]

    def get_edge_type_name(self, type_id: int) -> str:
        """Get the name for an edge type ID."""
        return self.edge_type_dict[type_id]

    def iter_nodes(self) -> Iterator[NodeData]:
        """Iterate over all nodes.

        Yields:
            NodeData objects for each node
        """
        for node_id in range(self.num_nodes):
            type_id = int(self.node_types[node_id].item())
            type_name = self.get_node_type_name(type_id)
            info = self.node_info.get(node_id, {})

            yield NodeData(
                node_id=node_id,
                node_type=type_name,
                node_type_id=type_id,
                info=info,
            )

    def iter_edges(self) -> Iterator[EdgeData]:
        """Iterate over all edges.

        Yields:
            EdgeData objects for each edge
        """
        src_ids = self.edge_index[0]
        dst_ids = self.edge_index[1]

        for edge_idx in range(self.num_edges):
            src_id = int(src_ids[edge_idx].item())
            dst_id = int(dst_ids[edge_idx].item())
            type_id = int(self.edge_types[edge_idx].item())
            type_name = self.get_edge_type_name(type_id)

            # Get node types for src and dst
            src_type_id = int(self.node_types[src_id].item())
            dst_type_id = int(self.node_types[dst_id].item())

            yield EdgeData(
                src_id=src_id,
                dst_id=dst_id,
                edge_type=type_name,
                edge_type_id=type_id,
                src_type=self.get_node_type_name(src_type_id),
                dst_type=self.get_node_type_name(dst_type_id),
            )

    def get_nodes_by_type(self, type_name: str) -> Iterator[NodeData]:
        """Get all nodes of a specific type.

        Args:
            type_name: The node type name

        Yields:
            NodeData objects for nodes of the given type
        """
        # Find the type ID for this name
        type_id = None
        for tid, tname in self.node_type_dict.items():
            if tname == type_name:
                type_id = tid
                break

        if type_id is None:
            raise ValueError(f"Unknown node type: {type_name}")

        for node_id in range(self.num_nodes):
            if int(self.node_types[node_id].item()) == type_id:
                info = self.node_info.get(node_id, {})
                yield NodeData(
                    node_id=node_id,
                    node_type=type_name,
                    node_type_id=type_id,
                    info=info,
                )

    def get_edges_by_type(self, type_name: str) -> Iterator[EdgeData]:
        """Get all edges of a specific type.

        Args:
            type_name: The edge type name

        Yields:
            EdgeData objects for edges of the given type
        """
        # Find the type ID for this name
        type_id = None
        for tid, tname in self.edge_type_dict.items():
            if tname == type_name:
                type_id = tid
                break

        if type_id is None:
            raise ValueError(f"Unknown edge type: {type_name}")

        src_ids = self.edge_index[0]
        dst_ids = self.edge_index[1]

        for edge_idx in range(self.num_edges):
            if int(self.edge_types[edge_idx].item()) != type_id:
                continue

            src_id = int(src_ids[edge_idx].item())
            dst_id = int(dst_ids[edge_idx].item())

            src_type_id = int(self.node_types[src_id].item())
            dst_type_id = int(self.node_types[dst_id].item())

            yield EdgeData(
                src_id=src_id,
                dst_id=dst_id,
                edge_type=type_name,
                edge_type_id=type_id,
                src_type=self.get_node_type_name(src_type_id),
                dst_type=self.get_node_type_name(dst_type_id),
            )

    def print_stats(self) -> None:
        """Print statistics about the loaded data."""
        print(f"STaRK-Prime Data Statistics:")
        print(f"  Nodes: {self.num_nodes:,}")
        print(f"  Edges: {self.num_edges:,}")
        print(f"  Node types ({len(self.node_type_dict)}):")
        for type_id, type_name in sorted(self.node_type_dict.items()):
            count = (self.node_types == type_id).sum().item()
            print(f"    - {type_name}: {count:,}")
        print(f"  Edge types ({len(self.edge_type_dict)}):")
        for type_id, type_name in sorted(self.edge_type_dict.items()):
            count = (self.edge_types == type_id).sum().item()
            print(f"    - {type_name}: {count:,}")


if __name__ == "__main__":
    # Quick test
    loader = PrimeDataLoader()
    loader.print_stats()
