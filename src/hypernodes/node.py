"""Node decorator and Node class for wrapping functions in pipelines."""

import functools
import inspect
from typing import Any, Callable, Union

from hypernodes.hypernode import HyperNode


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
    ):
        """Initialize a Node wrapper around a function.

        Args:
            func: The function to wrap
            output_name: Name for the output of this function
            cache: Whether to cache this node's output (default: True)
        """
        self.func = func
        self.name = func.__name__
        self._output_name = output_name
        self.cache = cache

        sig = inspect.signature(func)
        self._root_args = tuple(sig.parameters.keys())

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


def node(
    output_name: Union[str, tuple, Callable, None] = None,
    cache: bool = True,
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
    """
    # Handle @node (without parentheses) - output_name will be the function
    if callable(output_name):
        func = output_name
        return Node(func, output_name=func.__name__, cache=cache)

    # Handle @node() or @node(output_name="...") - return a decorator
    def decorator(func: Callable) -> Node:
        """Wrap the function in a Node."""
        # Use function name if output_name not provided
        final_output_name = output_name if output_name is not None else func.__name__
        return Node(func, output_name=final_output_name, cache=cache)

    return decorator
