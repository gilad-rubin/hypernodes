"""Node decorator and Node class for wrapping functions in pipelines."""

import functools
import inspect
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from hypernodes.hypernode import HyperNode


def _is_pydantic_model(obj: Any) -> bool:
    """Check if an object is a Pydantic model instance."""
    # Check for Pydantic v2 first, then v1
    return hasattr(obj, "model_fields") or hasattr(obj, "__fields__")


def _extract_field(obj: Any, field_name: str) -> Any:
    """Extract a field from a Pydantic model or dataclass."""
    return getattr(obj, field_name)


class Node(HyperNode):
    """Wraps a function with pipeline metadata.

    Implements the HyperNode protocol through structural subtyping.

    A Node represents an atomic unit of computation in a pipeline. It stores
    the original function along with metadata about its inputs and outputs.

    Attributes:
        func: The original Python function
        output_name: The name of this node's output
        cache: Whether this node's output should be cached
        root_args: Tuple of parameter names extracted from function signature
    """

    def __init__(
        self,
        func: Callable,
        output_name: Union[str, tuple],
        cache: bool = True,
        extract: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize a Node wrapper around a function.

        Args:
            func: The function to wrap
            output_name: Name for the output of this function
            cache: Whether to cache this node's output (default: True)
            extract: Field extraction mapping. Maps source parameter names to lists
                of field names to extract. Example: {"doc": ["file_path", "type"]}
                allows calling node(doc=my_document) and it will extract doc.file_path
                and doc.type, passing them as file_path and type to the function.
        """
        self.func = func
        self.name = func.__name__
        self._output_name = output_name
        self.cache = cache
        self._extract = extract or {}

        sig = inspect.signature(func)
        func_params = tuple(sig.parameters.keys())

        # Compute root_args: what the node expects from the pipeline
        # If extraction is configured, the node expects the source objects, not the extracted fields
        if self._extract:
            # Root args = source params (from extract) + any non-extracted params
            extracted_fields = set()
            for field_list in self._extract.values():
                extracted_fields.update(field_list)

            # Keep params that aren't extracted fields, add source params
            non_extracted_params = [p for p in func_params if p not in extracted_fields]
            source_params = list(self._extract.keys())
            self._root_args = tuple(source_params + non_extracted_params)
            self._func_params = func_params  # Store original func params for __call__
        else:
            self._root_args = func_params
            self._func_params = func_params

        # Pre-compute and cache code hash to avoid expensive recomputation
        # This is computed once at node creation and persists through pickling
        from .cache import hash_code

        self._code_hash = hash_code(func)

        # Preserve function metadata
        functools.update_wrapper(self, func)
        
        # Specifically handle async functions - wrap __call__ if needed?
        # No, we want Node to be transparent. 
        # But we need to mark Node instance as async if the function is async
        # so that inspect.iscoroutinefunction(node) works?
        # No, Node is not a coroutine function itself, its __call__ invokes one.
        # But we want engines to detect async-ness.
        
        # Mark this instance as async-like if the wrapped function is async
        # This helps engines detect async nodes without digging too deep
        if inspect.iscoroutinefunction(func) or (hasattr(func, "__code__") and (func.__code__.co_flags & 0x80)):
             self._is_async = True
        else:
             self._is_async = False

    @property
    def is_async(self) -> bool:
        return self._is_async


    @property
    def output_name(self) -> Union[str, tuple]:
        """Get the output name(s) of this node.

        Returns:
            Output name(s) for this node
        """
        return self._output_name

    @property
    def root_args(self) -> tuple:
        """Get the input parameter names required by this node.

        Returns:
            Tuple of parameter names from function signature
        """
        return self._root_args

    @property
    def code_hash(self) -> str:
        """Get cached code hash for this node's function.

        The hash is computed once at node creation and cached for reuse.
        This avoids expensive inspect.getsource() calls on every execution.
        The cached value persists through pickling/unpickling.

        Returns:
            SHA256 hash of the function's source code and closure
        """
        return self._code_hash

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the wrapped function with given arguments.

        This allows Node instances to be called directly like functions,
        which is useful in generated code and when wrapping nodes.

        If extraction is configured, source objects will have their fields
        extracted before calling the underlying function.

        Args:
            *args: Positional arguments to pass to the wrapped function
            **kwargs: Keyword arguments to pass to the wrapped function

        Returns:
            The result of executing the wrapped function
        """
        if self._extract and kwargs:
            kwargs = self._apply_extraction(kwargs)
        return self.func(*args, **kwargs)

    def _apply_extraction(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields from source objects in kwargs.

        Args:
            kwargs: Input keyword arguments

        Returns:
            Transformed kwargs with extracted fields
        """
        result = {}

        for source_name, field_names in self._extract.items():
            if source_name in kwargs:
                source_obj = kwargs[source_name]
                for field_name in field_names:
                    result[field_name] = _extract_field(source_obj, field_name)
            # Don't include source object in result - it's consumed by extraction

        # Pass through any kwargs that aren't source objects
        for key, value in kwargs.items():
            if key not in self._extract:
                result[key] = value

        return result

    def with_extraction(self, **extraction_spec) -> "Node":
        """Create a new Node with field extraction configured.

        This allows adapting existing portable nodes to work with rich objects
        like Pydantic models or dataclasses.

        Args:
            **extraction_spec: Keyword arguments mapping source parameter names
                to lists of field names to extract.
                Example: with_extraction(doc=["file_path", "document_type"])

        Returns:
            A new Node with extraction configured

        Example:
            >>> @node(output_name="result")
            ... def process(file_path: str, doc_type: str) -> dict:
            ...     return {"path": file_path, "type": doc_type}
            ...
            >>> # Create adapted version for Document objects
            >>> adapted = process.with_extraction(doc=["file_path", "doc_type"])
            >>> adapted(doc=my_document)  # Extracts fields automatically
        """
        # Merge with any existing extraction
        merged_extract = {**self._extract, **extraction_spec}
        return Node(
            func=self.func,
            output_name=self._output_name,
            cache=self.cache,
            extract=merged_extract,
        )

    def __repr__(self) -> str:
        """Return string representation of the Node."""
        extract_str = f", extract={self._extract}" if self._extract else ""
        return f"Node({self.func.__name__}, output={self.output_name}{extract_str})"

    def __hash__(self) -> int:
        """Make Node hashable for use in networkx graphs."""
        return hash((self.func.__name__, self.output_name))

    def __eq__(self, other) -> bool:
        """Check equality based on function and output name."""
        if not isinstance(other, Node):
            return False
        return (
            self.func.__name__ == other.func.__name__
            and self.output_name == other.output_name
        )

    def __getstate__(self):
        """Custom pickle support to preserve code hash cache."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Custom unpickle support to restore code hash cache."""
        self.__dict__.update(state)
        # _code_hash is preserved through pickling


def node(
    output_name: Union[str, tuple, Callable, None] = None,
    cache: bool = True,
    extract: Optional[Dict[str, List[str]]] = None,
) -> Union[Node, Callable[[Callable], Node]]:
    """Decorator to create Node instances from functions.

    This decorator wraps a function in a Node, making it part of a pipeline.
    The decorated function's parameters define its dependencies, and the
    output_name defines what other nodes can depend on.

    Can be used with or without parentheses:
    - @node (uses function name as output_name)
    - @node() (uses function name as output_name)
    - @node(output_name="result") (uses specified output_name)

    Args:
        output_name: Name for the output of this function. If None, uses the function's name.
                     When used as @node without parentheses, this receives the function itself.
        cache: Whether to cache this node's output (default: True)
        extract: Field extraction mapping for working with Pydantic models or dataclasses.
                 Maps source parameter names to lists of field names to extract.
                 Example: {"doc": ["file_path", "document_type"]} allows calling the node
                 with doc=my_document and it will extract the fields automatically.

    Returns:
        Either a Node (if used without parentheses) or a decorator function

    Example:
        >>> @node(output_name="result")
        ... def add_one(x: int) -> int:
        ...     return x + 1
        ...
        >>> pipeline = Pipeline(nodes=[add_one])
        >>> result = pipeline.run(inputs={"x": 5})
        >>> assert result == {"result": 6}

        >>> @node  # Uses function name as output_name
        ... def double(x: int) -> int:
        ...     return x * 2
        ...
        >>> pipeline = Pipeline(nodes=[double])
        >>> result = pipeline.run(inputs={"x": 5})
        >>> assert result == {"double": 10}

        >>> # With field extraction from Pydantic/dataclass
        >>> @node(output_name="parsed", extract={"doc": ["file_path", "doc_type"]})
        ... def parse_file(file_path: str, doc_type: str) -> dict:
        ...     return {"path": file_path, "type": doc_type}
        ...
        >>> # Now accepts a Document object, extracts fields automatically
        >>> parse_file(doc=my_document)
    """
    # Handle @node (without parentheses) - output_name will be the function
    if callable(output_name):
        func = output_name
        return Node(func, output_name=func.__name__, cache=cache, extract=extract)

    # Handle @node() or @node(output_name="...") - return a decorator
    def decorator(func: Callable) -> Node:
        """Wrap the function in a Node."""
        # Use function name if output_name not provided
        final_output_name = output_name if output_name is not None else func.__name__
        return Node(func, output_name=final_output_name, cache=cache, extract=extract)

    return decorator
