"""Node decorator and Node class for wrapping functions in pipelines."""

import functools
import inspect
from typing import Any, Callable, Union

from .node_protocol import HyperNode


class Node(HyperNode):
    """Wraps a function with pipeline metadata.

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
    ):
        """Initialize a Node wrapper around a function.

        Args:
            func: The function to wrap
            output_name: Name for the output of this function
            cache: Whether to cache this node's output (default: True)
        """
        self.func = func
        self.output_name = output_name
        self.cache = cache

        sig = inspect.signature(func)
        self.root_args = tuple(sig.parameters.keys())

        # Pre-compute and cache code hash to avoid expensive recomputation
        # This is computed once at node creation and persists through pickling
        from .cache import hash_code

        self._code_hash = hash_code(func)

        # Preserve function metadata
        functools.update_wrapper(self, func)

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

        Args:
            *args: Positional arguments to pass to the wrapped function
            **kwargs: Keyword arguments to pass to the wrapped function

        Returns:
            The result of executing the wrapped function
        """
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        """Return string representation of the Node."""
        return f"Node({self.func.__name__}, output={self.output_name})"

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


def node(output_name: str, cache: bool = True) -> Callable[[Callable], Node]:
    """Decorator to create Node instances from functions.

    This decorator wraps a function in a Node, making it part of a pipeline.
    The decorated function's parameters define its dependencies, and the
    output_name defines what other nodes can depend on.

    Args:
        output_name: Name for the output of this function
        cache: Whether to cache this node's output (default: True)

    Returns:
        A decorator function that wraps a function in a Node

    Example:
        >>> @node(output_name="result")
        ... def add_one(x: int) -> int:
        ...     return x + 1
        ...
        >>> pipeline = Pipeline(nodes=[add_one])
        >>> result = pipeline.run(inputs={"x": 5})
        >>> assert result == {"result": 6}
    """

    def decorator(func: Callable) -> Node:
        """Wrap the function in a Node."""
        return Node(func, output_name=output_name, cache=cache)

    return decorator
