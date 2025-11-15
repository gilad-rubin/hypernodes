"""Caching system with content-addressed computation signatures."""

import hashlib
import inspect
import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .pipeline import Pipeline
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass


def compute_signature(
    code_hash: str, inputs_hash: str, deps_hash: str = "", env_hash: str = ""
) -> str:
    """Compute computation signature for cache key.

    sig(node) = hash(code_hash + env_hash + inputs_hash + deps_hash)

    Args:
        code_hash: Hash of function source code
        inputs_hash: Hash of input values
        deps_hash: Hash of upstream dependency signatures
        env_hash: Hash of environment (versions, config salt, etc.)

    Returns:
        Hexadecimal signature string
    """
    combined = f"{code_hash}:{env_hash}:{inputs_hash}:{deps_hash}"
    return hashlib.sha256(combined.encode()).hexdigest()


def hash_code(func) -> str:
    """Compute hash of function source code including closure variables.

    Args:
        func: Python function to hash

    Returns:
        Hexadecimal hash of source code and closure
    """
    components = []

    # Get source code
    try:
        source = inspect.getsource(func)
        components.append(source)
    except (OSError, TypeError):
        # Can't get source (e.g., built-in function, lambda)
        # Fall back to function name + module
        components.append(f"{func.__module__}.{func.__name__}")

    # Include closure variables if present
    if func.__closure__:
        for cell in func.__closure__:
            try:
                # Get cell contents and add to hash
                components.append(str(cell.cell_contents))
            except ValueError:
                # Cell is empty
                pass

    combined = "::".join(components)
    return hashlib.sha256(combined.encode()).hexdigest()


def compute_pipeline_code_hash(pipeline: "Pipeline") -> str:
    """Compute aggregated code hash for all nodes in a pipeline.

    Aggregates hashes from all inner nodes, handling both regular nodes
    and nested pipeline nodes.

    Args:
        pipeline: Pipeline whose code to hash

    Returns:
        SHA256 hex digest of aggregated code hashes
    """
    inner_code_hashes = []
    for inner_node in pipeline.graph.execution_order:
        if hasattr(inner_node, "pipeline"):
            # Nested PipelineNode - use its pipeline ID
            inner_code_hashes.append(inner_node.pipeline.id)
        elif hasattr(inner_node, "func"):
            # Regular node - hash its function
            inner_code_hashes.append(hash_code(inner_node.func))

    return hashlib.sha256("::".join(inner_code_hashes).encode()).hexdigest()


def hash_value(value: Any, depth: int = 0, max_depth: int = 10) -> str:
    """Compute deterministic hash of a value.

    Handles primitive types, dicts, lists, dataclasses, and custom objects.
    Automatically serializes public attributes (excluding private ones starting with '_').

    For stateful objects (marked with @stateful), uses __cache_key__()
    if available, otherwise falls back to class name (warning: this may
    cause cache misses if initialization params differ).

    Args:
        value: Value to hash
        depth: Current recursion depth (internal use)
        max_depth: Maximum depth for nested object serialization

    Returns:
        Hexadecimal hash string
    """
    # Check recursion depth to avoid infinite loops
    if depth > max_depth:
        # At max depth, use repr as fallback
        return hashlib.sha256(repr(value).encode()).hexdigest()

    # Check for custom cache key (highest priority)
    # StatefulWrapper always has __cache_key__(), so this handles both:
    # 1. Stateful objects (via wrapper's __cache_key__)
    # 2. Custom objects with their own __cache_key__()
    if hasattr(value, "__cache_key__"):
        return hashlib.sha256(value.__cache_key__().encode()).hexdigest()

    # Handle None
    if value is None:
        return hashlib.sha256(b"None").hexdigest()

    # Handle primitives
    if isinstance(value, (str, int, float, bool)):
        return hashlib.sha256(str(value).encode()).hexdigest()

    # Handle lists and tuples
    if isinstance(value, (list, tuple)):
        item_hashes = [hash_value(item, depth + 1, max_depth) for item in value]
        combined = ":".join(item_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()

    # Handle dicts
    if isinstance(value, dict):
        # Sort keys for determinism
        sorted_items = sorted(value.items())
        item_hashes = [
            f"{k}={hash_value(v, depth + 1, max_depth)}" for k, v in sorted_items
        ]
        combined = ":".join(item_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()

    # Handle dataclasses - serialize all fields
    if is_dataclass(value) and not isinstance(value, type):
        class_name = value.__class__.__name__
        field_hashes = []
        for field in dataclass_fields(value):
            field_value = getattr(value, field.name)
            field_hash = hash_value(field_value, depth + 1, max_depth)
            field_hashes.append(f"{field.name}={field_hash}")
        combined = f"{class_name}::{'::'.join(sorted(field_hashes))}"
        return hashlib.sha256(combined.encode()).hexdigest()

    # Handle custom classes - serialize public attributes only
    if hasattr(value, "__dict__"):
        class_name = value.__class__.__name__
        # Get public attributes (exclude private ones starting with '_')
        public_attrs = {
            k: v for k, v in value.__dict__.items() if not k.startswith("_")
        }

        if public_attrs:
            # Serialize public attributes
            sorted_items = sorted(public_attrs.items())
            attr_hashes = [
                f"{k}={hash_value(v, depth + 1, max_depth)}" for k, v in sorted_items
            ]
            combined = f"{class_name}::{'::'.join(attr_hashes)}"
            return hashlib.sha256(combined.encode()).hexdigest()

    # Fall back to pickle for other objects
    try:
        pickled = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(pickled).hexdigest()
    except Exception:
        # Last resort: use repr
        return hashlib.sha256(repr(value).encode()).hexdigest()


def hash_inputs(inputs: Dict[str, Any]) -> str:
    """Compute hash of input dictionary.

    Args:
        inputs: Dictionary of input values

    Returns:
        Hexadecimal hash string
    """
    return hash_value(inputs)


class Cache(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, signature: str) -> Optional[Any]:
        """Retrieve cached output by signature.

        Args:
            signature: Computation signature

        Returns:
            Cached output if found, None otherwise
        """
        pass

    @abstractmethod
    def put(self, signature: str, output: Any) -> None:
        """Store output with given signature.

        Args:
            signature: Computation signature
            output: Output value to cache
        """
        pass

    @abstractmethod
    def has(self, signature: str) -> bool:
        """Check if signature exists in cache.

        Args:
            signature: Computation signature

        Returns:
            True if signature exists, False otherwise
        """
        pass


class DiskCache(Cache):
    """Disk-based cache implementation.

    Uses filesystem for both MetaStore (signatures) and BlobStore (outputs).
    - MetaStore: JSON file mapping signatures to metadata
    - BlobStore: Pickle files for cached outputs

    Attributes:
        path: Root directory for cache storage
        meta_store: Dict mapping signatures to metadata
    """

    def __init__(self, path: str):
        """Initialize disk cache.

        Args:
            path: Directory path for cache storage
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        # BlobStore directory
        self.blob_dir = self.path / "blobs"
        self.blob_dir.mkdir(exist_ok=True)

        # MetaStore file
        self.meta_file = self.path / "meta.json"
        self.meta_store = self._load_meta()

    def _load_meta(self) -> Dict[str, Dict]:
        """Load metadata from disk."""
        if self.meta_file.exists():
            with open(self.meta_file, "r") as f:
                return json.load(f)
        return {}

    def _save_meta(self) -> None:
        """Save metadata to disk."""
        with open(self.meta_file, "w") as f:
            json.dump(self.meta_store, f, indent=2)

    def _blob_path(self, signature: str) -> Path:
        """Get path to blob file for signature."""
        return self.blob_dir / f"{signature}.pkl"

    def get(self, signature: str) -> Optional[Any]:
        """Retrieve cached output by signature."""
        if not self.has(signature):
            return None

        # Load from BlobStore
        blob_path = self._blob_path(signature)
        if not blob_path.exists():
            return None

        with open(blob_path, "rb") as f:
            return pickle.load(f)

    def put(self, signature: str, output: Any) -> None:
        """Store output with given signature."""
        # Save to BlobStore
        blob_path = self._blob_path(signature)
        with open(blob_path, "wb") as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update MetaStore
        self.meta_store[signature] = {
            "blob_path": str(blob_path.relative_to(self.path))
        }
        self._save_meta()

    def has(self, signature: str) -> bool:
        """Check if signature exists in cache."""
        return signature in self.meta_store

    def clear(self) -> None:
        """Clear all cached data."""
        # Remove all blob files
        import shutil

        if self.blob_dir.exists():
            shutil.rmtree(self.blob_dir)
            self.blob_dir.mkdir()

        # Clear metadata
        self.meta_store = {}
        self._save_meta()
