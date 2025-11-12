"""Mixins for reusable behavior across Hypernodes classes."""

from typing import TYPE_CHECKING, List, Self

if TYPE_CHECKING:
    from hypernodes.cache import Cache
    from hypernodes.callbacks import PipelineCallback
    from hypernodes.engine import Engine


class BuilderMixin:
    """Mixin providing fluent builder methods for pipeline configuration.

    Classes using this mixin must have these attributes:
    - engine
    - cache
    - callbacks
    - name
    """

    def with_engine(self, engine: "Engine") -> Self:
        """Configure with a specific engine.

        Args:
            engine: Engine instance (HypernodesEngine, DaftEngine, etc.)

        Returns:
            Self for method chaining
        """
        self.engine = engine
        return self

    def with_cache(self, cache: "Cache") -> Self:
        """Configure with a cache backend.

        Args:
            cache: Cache instance (DiskCache, etc.)

        Returns:
            Self for method chaining
        """
        self.cache = cache
        return self

    def with_callbacks(self, callbacks: List["PipelineCallback"]) -> Self:
        """Configure with pipeline callbacks.

        Args:
            callbacks: List of callback instances

        Returns:
            Self for method chaining
        """
        self.callbacks = callbacks
        return self

    def with_name(self, name: str) -> Self:
        """Set the pipeline name.

        Args:
            name: Name for the pipeline

        Returns:
            Self for method chaining
        """
        self.name = name
        return self

    def with_backend(self, backend: "Engine") -> Self:
        """Configure with a backend (deprecated alias for with_engine).

        Args:
            backend: Engine instance

        Returns:
            Self for method chaining
        """
        return self.with_engine(backend)
