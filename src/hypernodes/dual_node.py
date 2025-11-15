"""DualNode: A node that supports both singular and batch execution modes.

DualNode allows you to define two implementations:
- singular: For single-item execution (used in .run())
- batch: For batch execution (used in .map())

This enables type-safe, scalar-first design while maintaining batch optimization.
"""

import inspect
from typing import Any, Callable, Optional, Tuple, Union

from .hypernode import HyperNode


class DualNode:
    """A node with dual execution modes: singular and batch.

    The singular function defines the canonical signature (used for type hints,
    visualization, and single-item execution). The batch function provides an
    optimized implementation for processing multiple items at once.

    Examples:
        Stateless DualNode:
        >>> def encode_one(text: str, encoder: Encoder) -> list[float]:
        ...     return encoder.encode(text)
        ...
        >>> def encode_many(texts: Series[str], encoder: Encoder) -> Series[list[float]]:
        ...     return encoder.encode_batch(texts)
        ...
        >>> node = DualNode(
        ...     output_name="encoded",
        ...     singular=encode_one,
        ...     batch=encode_many
        ... )

        Stateful DualNode:
        >>> class TextOps:
        ...     def __init__(self, model_name: str):
        ...         self.model = load_model(model_name)
        ...
        ...     def process_one(self, text: str) -> str:
        ...         return self.model.process(text)
        ...
        ...     def process_many(self, texts: Series[str]) -> Series[str]:
        ...         return self.model.process_batch(texts)
        ...
        >>> ops = TextOps(model_name="my-model")
        >>> node = DualNode(
        ...     output_name="processed",
        ...     singular=ops.process_one,
        ...     batch=ops.process_many
        ... )
    """

    def __init__(
        self,
        output_name: Union[str, Tuple[str, ...]],
        singular: Callable,
        batch: Callable,
        cache: bool = True,
    ):
        """Initialize a DualNode.

        Args:
            output_name: Name(s) of output(s) produced by this node
            singular: Function for single-item execution (defines canonical signature)
            batch: Function for batch execution (optimized for multiple items)
            cache: Whether to cache this node's outputs (default: True)
        """
        self.output_name = output_name
        self.singular = singular
        self.batch = batch
        self.cache = cache
        self.is_dual_node = True  # Flag for engine detection

        # Extract root_args from singular function (canonical signature)
        sig = inspect.signature(singular)
        self.root_args = tuple(sig.parameters.keys())

        # Compute code hash from both functions
        self._code_hash = None

        # Detect if stateful (bound methods)
        self.is_stateful = self._detect_stateful()

        # Store instance reference for stateful nodes
        self.instance = None
        if self.is_stateful:
            # Extract instance from bound method
            if hasattr(singular, "__self__"):
                self.instance = singular.__self__

    def _detect_stateful(self) -> bool:
        """Detect if this is a stateful node (uses bound methods)."""
        return (
            hasattr(self.singular, "__self__")
            or hasattr(self.batch, "__self__")
        )

    @property
    def code_hash(self) -> str:
        """Compute code hash from both singular and batch functions.

        This ensures cache invalidation when either implementation changes.
        """
        if self._code_hash is None:
            import hashlib

            # Hash both function implementations
            singular_code = self._get_function_code(self.singular)
            batch_code = self._get_function_code(self.batch)

            combined = f"{singular_code}|{batch_code}".encode("utf-8")
            self._code_hash = hashlib.sha256(combined).hexdigest()

        return self._code_hash

    def _get_function_code(self, func: Callable) -> str:
        """Extract source code from function or method."""
        try:
            # Handle bound methods
            if hasattr(func, "__func__"):
                func = func.__func__

            return inspect.getsource(func)
        except (OSError, TypeError):
            # Fallback: use function name and qualname
            return f"{func.__module__}.{func.__qualname__}"

    @property
    def name(self) -> str:
        """Return node name for identification.
        
        Strips common suffixes (_singular, _one, _single) to get clean base name.
        """
        # Use singular function name (canonical)
        if hasattr(self.singular, "__name__"):
            func_name = self.singular.__name__
            
            # Strip common suffixes to get clean name
            for suffix in ["_singular", "_one", "_single"]:
                if func_name.endswith(suffix):
                    return func_name[: -len(suffix)]
            
            return func_name
        return str(id(self))
    
    @property
    def func(self):
        """Expose singular function as 'func' for compatibility with visualization.
        
        This allows visualization tools to extract type hints from the singular function.
        """
        return self.singular

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DualNode("
            f"name={self.name}, "
            f"output={self.output_name}, "
            f"stateful={self.is_stateful})"
        )

