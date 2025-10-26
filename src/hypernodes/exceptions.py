"""Custom exceptions for HyperNodes pipeline system."""


class HyperNodesError(Exception):
    """Base exception for all HyperNodes errors."""
    pass


class CycleError(HyperNodesError):
    """Raised when a cycle is detected in the pipeline DAG."""
    pass


class DependencyError(HyperNodesError):
    """Raised when a dependency cannot be satisfied."""
    pass


class ExecutionError(HyperNodesError):
    """Raised when a node execution fails."""
    pass
