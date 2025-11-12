"""Execution context for pipeline runs.

This module provides the ExecutionContext class which encapsulates
configuration and state for pipeline execution, replacing the _ctx parameter.
"""

from typing import TYPE_CHECKING, Optional

from .callbacks import CallbackContext

if TYPE_CHECKING:
    from .pipeline_config import PipelineConfiguration


class ExecutionContext:
    """Manages execution state and configuration for pipeline runs.
    
    Replaces the _ctx parameter with a proper context object that encapsulates
    both configuration and callback state.
    
    Attributes:
        config: Pipeline configuration (engine, cache, callbacks, name)
        callback_context: State for callback tracking (nested pipelines, progress, etc.)
    """
    
    def __init__(self, config: "PipelineConfiguration", callback_context: Optional[CallbackContext] = None):
        """Initialize execution context.
        
        Args:
            config: Pipeline configuration
            callback_context: Optional callback context (created if not provided)
        """
        self.config = config
        self.callback_context = callback_context if callback_context is not None else CallbackContext()
    
    def for_nested_pipeline(self, nested_config: "PipelineConfiguration") -> "ExecutionContext":
        """Create child context for nested pipeline execution.
        
        Merges nested pipeline config with parent config (child takes precedence).
        Shares the same callback context for unified progress tracking.
        
        Args:
            nested_config: Configuration from nested pipeline
            
        Returns:
            New ExecutionContext with merged config and shared callback context
            
        Example:
            >>> parent_ctx = ExecutionContext(parent_config)
            >>> child_ctx = parent_ctx.for_nested_pipeline(child_config)
            >>> # child_ctx.config has merged settings
            >>> # child_ctx.callback_context is same object (shared state)
        """
        merged_config = nested_config.merge_with(self.config)
        return ExecutionContext(merged_config, self.callback_context)
    
    @property
    def engine(self):
        """Convenience property for accessing effective engine."""
        return self.config.effective_engine
    
    @property
    def cache(self):
        """Convenience property for accessing effective cache."""
        return self.config.effective_cache
    
    @property
    def callbacks(self):
        """Convenience property for accessing effective callbacks."""
        return self.config.effective_callbacks
