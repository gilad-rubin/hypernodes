"""Async execution strategies for HyperNodes.

This module provides strategies for handling async execution in sync contexts.
"""

from typing import TYPE_CHECKING, Any

from .executors import AsyncExecutor

if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .protocols import AsyncStrategy


class PerCallAsyncStrategy:
    """Per-call async strategy (status quo)."""

    def should_use_async(
        self, pipeline: "Pipeline", node_executor: Any
    ) -> tuple[bool, str]:
        """Never use native async path."""
        return False, "per_call"


class ThreadLocalAsyncStrategy:
    """Thread-local async strategy."""

    def should_use_async(
        self, pipeline: "Pipeline", node_executor: Any
    ) -> tuple[bool, str]:
        """Never use native async path (thread-local handling in sync context)."""
        return False, "thread_local"


class AsyncNativeStrategy:
    """Async-native strategy (end-to-end async when possible)."""

    def should_use_async(
        self, pipeline: "Pipeline", node_executor: Any
    ) -> tuple[bool, str]:
        """Use async if executor is AsyncExecutor and pipeline supports it."""
        from .engine_orchestrator import _pipeline_supports_async_native

        if not isinstance(node_executor, AsyncExecutor):
            return False, "thread_local"

        if _pipeline_supports_async_native(pipeline):
            return True, "thread_local"

        return False, "thread_local"


class AutoAsyncStrategy:
    """Auto async strategy (hybrid detection)."""

    def should_use_async(
        self, pipeline: "Pipeline", node_executor: Any
    ) -> tuple[bool, str]:
        """Use async if executor is AsyncExecutor and pipeline supports it."""
        from .engine_orchestrator import _pipeline_supports_async_native

        if not isinstance(node_executor, AsyncExecutor):
            return False, "thread_local"

        if _pipeline_supports_async_native(pipeline):
            return True, "thread_local"

        return False, "thread_local"


def create_async_strategy(strategy_name: str) -> "AsyncStrategy":
    """Factory for async strategies.

    Args:
        strategy_name: One of "per_call", "thread_local", "async_native", "auto"

    Returns:
        AsyncStrategy instance

    Raises:
        ValueError: If strategy_name is invalid
    """
    strategies = {
        "per_call": PerCallAsyncStrategy(),
        "thread_local": ThreadLocalAsyncStrategy(),
        "async_native": AsyncNativeStrategy(),
        "auto": AutoAsyncStrategy(),
    }

    strategy = strategies.get(strategy_name)
    if strategy is None:
        raise ValueError(
            f"Invalid async_strategy: {strategy_name}. "
            f"Must be one of {list(strategies.keys())}"
        )

    return strategy
