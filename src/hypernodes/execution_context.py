"""Execution context for pipeline runs.

This module provides the ExecutionContext class which encapsulates
configuration and state for pipeline execution.

It also provides a ContextVar for passing CallbackContext through the call stack
without polluting function signatures.
"""

from contextvars import ContextVar
from typing import TYPE_CHECKING, List, Optional

from .callbacks import CallbackContext

if TYPE_CHECKING:
    from .cache import Cache
    from .callbacks import PipelineCallback
    from .engine import Engine

# ContextVar for passing CallbackContext through execution chain
# This allows nested pipelines to share callback context without
# polluting public API or causing state pollution on Pipeline instances
_callback_context: ContextVar[Optional[CallbackContext]] = ContextVar(
    'hypernodes_callback_context', default=None
)


def get_callback_context() -> Optional[CallbackContext]:
    """Get the current CallbackContext from the execution context.
    
    Returns:
        Current CallbackContext if set, None otherwise
    """
    return _callback_context.get()


def set_callback_context(ctx: Optional[CallbackContext]) -> None:
    """Set the CallbackContext for the current execution context.
    
    Args:
        ctx: CallbackContext to set, or None to clear
    """
    _callback_context.set(ctx)


class ExecutionContext:
    """Manages execution state and configuration for pipeline runs.
    
    Encapsulates both configuration (engine, cache, callbacks) and runtime
    state (callback context) for pipeline execution. Handles configuration
    inheritance for nested pipelines.
    
    Attributes:
        engine: Execution engine (or None to use default)
        cache: Cache backend (or None for no caching)
        callbacks: List of callback instances
        callback_context: State for callback tracking (nested pipelines, progress, etc.)
    """
    
    def __init__(
        self,
        engine: Optional["Engine"] = None,
        cache: Optional["Cache"] = None,
        callbacks: Optional[List["PipelineCallback"]] = None,
        callback_context: Optional[CallbackContext] = None
    ):
        """Initialize execution context.
        
        Args:
            engine: Execution engine (None to use default)
            cache: Cache backend (None for no caching)
            callbacks: List of callback instances
            callback_context: Optional callback context (created if not provided)
        """
        self.engine = engine
        self.cache = cache
        self.callbacks = callbacks
        self.callback_context = callback_context if callback_context is not None else CallbackContext()
    
    @classmethod
    def from_pipeline(cls, pipeline, callback_context: Optional[CallbackContext] = None) -> "ExecutionContext":
        """Create execution context from a pipeline's configuration.
        
        Args:
            pipeline: Pipeline or PipelineNode to extract configuration from
            callback_context: Optional callback context (created if not provided)
            
        Returns:
            New ExecutionContext with pipeline's configuration
        """
        return cls(
            engine=getattr(pipeline, 'engine', None) or getattr(pipeline, '_engine', None),
            cache=getattr(pipeline, 'cache', None) or getattr(pipeline, '_cache', None),
            callbacks=getattr(pipeline, 'callbacks', None) or getattr(pipeline, '_callbacks', None),
            callback_context=callback_context
        )
    
    def merge_with(self, parent_ctx: Optional["ExecutionContext"]) -> "ExecutionContext":
        """Merge with parent context (self takes precedence).
        
        Creates a new context that inherits from parent where this context
        doesn't specify values. Callback context is shared between parent and child.
        
        Args:
            parent_ctx: Parent context to inherit from (can be None)
            
        Returns:
            New ExecutionContext with merged configuration and shared callback context
            
        Example:
            >>> parent = ExecutionContext(engine=my_engine, cache=my_cache)
            >>> child = ExecutionContext(cache=other_cache)
            >>> merged = child.merge_with(parent)
            >>> # merged.engine == my_engine (inherited from parent)
            >>> # merged.cache == other_cache (child overrides)
            >>> # merged.callback_context is shared with parent
        """
        if parent_ctx is None:
            return ExecutionContext(
                engine=self.engine,
                cache=self.cache,
                callbacks=self.callbacks,
                callback_context=self.callback_context
            )
        
        return ExecutionContext(
            engine=self.engine if self.engine is not None else parent_ctx.engine,
            cache=self.cache if self.cache is not None else parent_ctx.cache,
            callbacks=self.callbacks if self.callbacks is not None else parent_ctx.callbacks,
            callback_context=parent_ctx.callback_context  # Always share with parent
        )
    
    def for_nested_pipeline(self, nested_pipeline) -> "ExecutionContext":
        """Create child context for nested pipeline execution.
        
        Merges nested pipeline config with parent config (child takes precedence).
        Shares the same callback context for unified progress tracking.
        
        Args:
            nested_pipeline: Pipeline or PipelineNode to execute
            
        Returns:
            New ExecutionContext with merged config and shared callback context
            
        Example:
            >>> parent_ctx = ExecutionContext.from_pipeline(parent_pipeline)
            >>> child_ctx = parent_ctx.for_nested_pipeline(child_pipeline)
            >>> # child_ctx has merged settings
            >>> # child_ctx.callback_context is same object (shared state)
        """
        nested_ctx = ExecutionContext.from_pipeline(nested_pipeline, self.callback_context)
        return nested_ctx.merge_with(self)
    
    @property
    def effective_engine(self) -> "Engine":
        """Get effective engine with default fallback.
        
        Returns:
            Configured engine or default HypernodesEngine instance
        """
        if self.engine is not None:
            return self.engine
        
        # Import here to avoid circular dependency
        from .engine import HypernodesEngine
        return HypernodesEngine()
    
    @property
    def effective_cache(self) -> Optional["Cache"]:
        """Get effective cache (may be None for no caching).
        
        Returns:
            Configured cache or None
        """
        return self.cache
    
    @property
    def effective_callbacks(self) -> List["PipelineCallback"]:
        """Get effective callbacks with empty list fallback.
        
        Returns:
            Configured callbacks or empty list
        """
        return self.callbacks or []
