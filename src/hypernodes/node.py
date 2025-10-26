"""Node decorator and Node class for wrapping functions in pipelines."""
import functools
import inspect
from typing import Callable, Any


class Node:
    """Wraps a function with pipeline metadata.
    
    A Node represents an atomic unit of computation in a pipeline. It stores
    the original function along with metadata about its inputs and outputs.
    
    Attributes:
        func: The original Python function
        output_name: The name of this node's output
        cache: Whether this node's output should be cached
        parameters: Tuple of parameter names extracted from function signature
    """
    
    def __init__(
        self,
        func: Callable,
        output_name: str,
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
        
        # Extract parameters using inspect (inspired by pipefunc)
        sig = inspect.signature(func)
        self.parameters = tuple(sig.parameters.keys())
        
        # Preserve function metadata
        functools.update_wrapper(self, func)
    
    def __call__(self, **kwargs) -> Any:
        """Execute the wrapped function with given keyword arguments.
        
        Args:
            **kwargs: Keyword arguments matching the function's parameters
            
        Returns:
            The result of executing the wrapped function
        """
        return self.func(**kwargs)
    
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
        return (self.func.__name__ == other.func.__name__ and 
                self.output_name == other.output_name)


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
