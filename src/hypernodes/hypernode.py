"""Protocol for executable nodes in a pipeline."""

from typing import Protocol, Union


class HyperNode(Protocol):
    """Protocol for executable units in a pipeline.

    Both Node (wrapping functions) and PipelineNode (wrapping pipelines)
    implement this protocol, providing a consistent interface for their
    inputs, outputs, and caching behavior.
    
    Uses structural subtyping (Protocol) rather than nominal subtyping (ABC),
    allowing flexibility in implementation while maintaining type safety.
    """

    name: str
    """Display name for this node (function name or user-provided)."""

    cache: bool
    """Whether this node's output should be cached."""

    @property
    def root_args(self) -> tuple:
        """Return tuple of input parameter names required by this node."""
        ...

    @property
    def output_name(self) -> Union[str, tuple]:
        """Return output name(s) produced by this node.

        Returns:
            str for single output, tuple for multiple outputs
        """
        ...

    @property
    def code_hash(self) -> str:
        """Return hash of the node's code for cache invalidation.

        For Node: hash of the wrapped function's source code
        For PipelineNode: aggregated hash of all inner nodes

        Returns:
            SHA256 hex digest string
        """
        ...
