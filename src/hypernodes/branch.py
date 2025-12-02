"""Branch node for conditional execution routing.

Branch nodes are decision points that route execution based on a boolean condition.
They produce mutually-exclusive "gate signals" that become implicit dependencies
for target nodes, ensuring only the winning path executes.

Example:
    >>> from hypernodes import branch, node, Pipeline
    >>>
    >>> @node(output_name="value")
    >>> def get_value(x: int) -> int:
    ...     return x
    >>>
    >>> @branch(when_true=process_positive, when_false=process_negative)
    >>> def is_positive(value: int) -> bool:
    ...     return value > 0
    >>>
    >>> @node(output_name="result")
    >>> def process_positive(value: int) -> str:
    ...     return f"Positive: {value}"
    >>>
    >>> @node(output_name="result")
    >>> def process_negative(value: int) -> str:
    ...     return f"Non-positive: {value}"
    >>>
    >>> pipeline = Pipeline(nodes=[get_value, is_positive, process_positive, process_negative])
    >>> result = pipeline.run(inputs={"x": 5})
    >>> print(result)  # {"value": 5, "result": "Positive: 5"}
"""

import functools
import inspect
from typing import Any, Callable, Union

from .hypernode import HyperNode


class BranchNode(HyperNode):
    """A node that routes execution based on a boolean condition.

    BranchNode wraps a function that returns a boolean. Based on the result,
    execution continues to either the `when_true` or `when_false` target.

    The branch produces two mutually-exclusive gate signals:
    - `_gate_{name}_true`: Produced when the function returns True
    - `_gate_{name}_false`: Produced when the function returns False

    Target nodes implicitly depend on these gate signals, so only the
    winning path executes.

    Attributes:
        func: The wrapped boolean function
        when_true: Target node/function for True branch
        when_false: Target node/function for False branch
        name: Function name (used in visualization and debugging)
        cache: Always False - branch decisions should be re-evaluated
    """

    def __init__(
        self,
        func: Callable[..., bool],
        when_true: Union[Callable, "HyperNode"],
        when_false: Union[Callable, "HyperNode"],
    ):
        """Initialize a BranchNode.

        Args:
            func: Function that returns a boolean for routing
            when_true: Target node/function when condition is True
            when_false: Target node/function when condition is False
        """
        self.func = func
        self.when_true = when_true
        self.when_false = when_false
        self.name = func.__name__

        # Branch nodes should not be cached - decisions should be re-evaluated
        self.cache = False

        # Extract parameter names from function signature
        sig = inspect.signature(func)
        self._root_args = tuple(sig.parameters.keys())

        # Pre-compute code hash
        from .cache import hash_code

        self._code_hash = hash_code(func)

        # Preserve function metadata
        functools.update_wrapper(self, func)

        # Mark as branch node for detection
        self._is_branch = True

    @property
    def output_name(self) -> tuple:
        """Return gate signal names for True and False branches.

        Returns:
            Tuple of (true_gate_name, false_gate_name)
        """
        return (f"_gate_{self.name}_true", f"_gate_{self.name}_false")

    @property
    def true_gate(self) -> str:
        """Return the True gate signal name."""
        return self.output_name[0]

    @property
    def false_gate(self) -> str:
        """Return the False gate signal name."""
        return self.output_name[1]

    @property
    def root_args(self) -> tuple:
        """Return input parameter names required by this branch.

        Returns:
            Tuple of parameter names from function signature
        """
        return self._root_args

    @property
    def code_hash(self) -> str:
        """Return cached code hash for this branch's function.

        Returns:
            SHA256 hash of the function's source code
        """
        return self._code_hash

    @property
    def when_true_name(self) -> str:
        """Return the name of the True target node."""
        return self._get_target_name(self.when_true)

    @property
    def when_false_name(self) -> str:
        """Return the name of the False target node."""
        return self._get_target_name(self.when_false)

    def _get_target_name(self, target: Union[Callable, "HyperNode"]) -> str:
        """Extract the name from a target node or function.

        Args:
            target: Node instance or function reference

        Returns:
            The name of the target
        """
        if hasattr(target, "name"):
            return target.name
        if hasattr(target, "__name__"):
            return target.__name__
        return str(target)

    def __call__(self, *args, **kwargs) -> bool:
        """Execute the branch function and return the boolean result.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Boolean result of the branch condition
        """
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        """Return string representation of the BranchNode."""
        return (
            f"BranchNode({self.name}, "
            f"when_true={self.when_true_name}, "
            f"when_false={self.when_false_name})"
        )

    def __hash__(self) -> int:
        """Make BranchNode hashable for use in graphs."""
        return hash((self.name, self.output_name))

    def __eq__(self, other) -> bool:
        """Check equality based on function and targets."""
        if not isinstance(other, BranchNode):
            return False
        return (
            self.name == other.name
            and self.when_true_name == other.when_true_name
            and self.when_false_name == other.when_false_name
        )


def branch(
    when_true: Union[Callable, HyperNode],
    when_false: Union[Callable, HyperNode],
) -> Callable[[Callable[..., bool]], BranchNode]:
    """Decorator to create a BranchNode for conditional execution routing.

    The decorated function should return a boolean. Based on the result:
    - True: Execution continues at `when_true` target
    - False: Execution continues at `when_false` target

    Target nodes in the non-selected branch will be skipped (not executed).

    Args:
        when_true: Target node/function when condition returns True
        when_false: Target node/function when condition returns False

    Returns:
        Decorator function that creates a BranchNode

    Example:
        >>> @branch(when_true=process_valid, when_false=handle_invalid)
        ... def is_valid(data: dict) -> bool:
        ...     return data.get("valid", False)
    """

    def decorator(func: Callable[..., bool]) -> BranchNode:
        """Wrap the function in a BranchNode."""
        return BranchNode(func, when_true=when_true, when_false=when_false)

    return decorator

