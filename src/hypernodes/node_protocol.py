"""Abstract base class for executable nodes in a pipeline."""

from abc import ABC, abstractmethod
from typing import Union


class HyperNode(ABC):
    """Abstract interface for executable units in a pipeline.
    
    Both Node (wrapping functions) and PipelineNode (wrapping pipelines)
    implement this interface, providing a consistent way to query their
    inputs, outputs, and caching behavior.
    """

    @property
    @abstractmethod
    def root_args(self) -> tuple:
        """Return tuple of input parameter names required by this node."""
        ...

    @property
    @abstractmethod
    def output_name(self) -> Union[str, tuple]:
        """Return output name(s) produced by this node.
        
        Returns:
            str for single output, tuple for multiple outputs
        """
        ...

    @property
    @abstractmethod
    def cache(self) -> bool:
        """Return whether this node's output should be cached."""
        ...
