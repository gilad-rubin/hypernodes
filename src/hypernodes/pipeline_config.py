"""Pipeline configuration management.

This module provides the PipelineConfiguration class for managing
pipeline execution settings (engine, cache, callbacks, name).

Uses composition pattern instead of parent tracking for configuration inheritance.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .cache import Cache
    from .callbacks import PipelineCallback
    from .engine import Engine


@dataclass
class PipelineConfiguration:
    """Encapsulates pipeline execution configuration.
    
    Handles configuration with composition pattern instead of parent tracking.
    Configuration can be merged with parent config where child values take precedence.
    
    Attributes:
        engine: Execution engine (HypernodesEngine, DaftEngine, etc.)
        cache: Cache backend for result caching (DiskCache, etc.)
        callbacks: List of callback instances for lifecycle hooks
        name: Human-readable name for the pipeline
    """
    
    engine: Optional["Engine"] = None
    cache: Optional["Cache"] = None
    callbacks: Optional[List["PipelineCallback"]] = None
    name: Optional[str] = None
    
    def merge_with(self, parent_config: Optional["PipelineConfiguration"]) -> "PipelineConfiguration":
        """Merge with parent configuration, child values take precedence.
        
        Creates a new configuration that inherits from parent where
        this configuration doesn't specify values.
        
        Args:
            parent_config: Parent configuration to inherit from (can be None)
            
        Returns:
            New PipelineConfiguration with merged values
            
        Example:
            >>> parent = PipelineConfiguration(engine=my_engine, cache=my_cache)
            >>> child = PipelineConfiguration(cache=other_cache)
            >>> merged = child.merge_with(parent)
            >>> # merged.engine == my_engine (inherited from parent)
            >>> # merged.cache == other_cache (child overrides)
        """
        if parent_config is None:
            return PipelineConfiguration(
                engine=self.engine,
                cache=self.cache,
                callbacks=self.callbacks,
                name=self.name
            )
        
        return PipelineConfiguration(
            engine=self.engine if self.engine is not None else parent_config.engine,
            cache=self.cache if self.cache is not None else parent_config.cache,
            callbacks=self.callbacks if self.callbacks is not None else parent_config.callbacks,
            name=self.name  # Name is never inherited
        )
    
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
    
    def with_engine(self, engine: "Engine") -> "PipelineConfiguration":
        """Create new config with updated engine.
        
        Args:
            engine: Engine instance to use
            
        Returns:
            New PipelineConfiguration with updated engine
        """
        return PipelineConfiguration(
            engine=engine,
            cache=self.cache,
            callbacks=self.callbacks,
            name=self.name
        )
    
    def with_cache(self, cache: "Cache") -> "PipelineConfiguration":
        """Create new config with updated cache.
        
        Args:
            cache: Cache instance to use
            
        Returns:
            New PipelineConfiguration with updated cache
        """
        return PipelineConfiguration(
            engine=self.engine,
            cache=cache,
            callbacks=self.callbacks,
            name=self.name
        )
    
    def with_callbacks(self, callbacks: List["PipelineCallback"]) -> "PipelineConfiguration":
        """Create new config with updated callbacks.
        
        Args:
            callbacks: List of callback instances
            
        Returns:
            New PipelineConfiguration with updated callbacks
        """
        return PipelineConfiguration(
            engine=self.engine,
            cache=self.cache,
            callbacks=callbacks,
            name=self.name
        )
    
    def with_name(self, name: str) -> "PipelineConfiguration":
        """Create new config with updated name.
        
        Args:
            name: Human-readable pipeline name
            
        Returns:
            New PipelineConfiguration with updated name
        """
        return PipelineConfiguration(
            engine=self.engine,
            cache=self.cache,
            callbacks=self.callbacks,
            name=name
        )
