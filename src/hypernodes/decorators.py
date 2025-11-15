"""Engine-agnostic decorators for HyperNodes."""

import hashlib


def stateful(cls: type) -> type:
    """Mark a class as stateful with lazy initialization.

    Creates a wrapper that delays expensive __init__ until first use.
    This enables:
    - Lazy initialization: __init__ args stored, not called until needed
    - Efficient serialization: only init args pickled, not heavy state
    - Engine optimization: SequentialEngine caches, DaftEngine uses @daft.cls

    Example:
        >>> from hypernodes import stateful, node, Pipeline
        >>>
        >>> @stateful
        ... class ExpensiveModel:
        ...     def __init__(self, model_path: str):
        ...         self.model = load_model(model_path)  # Lazy - only when needed
        ...
        ...     def predict(self, text: str) -> str:
        ...         return self.model(text)
        >>>
        >>> # Creating instance doesn't call __init__ yet
        >>> model = ExpensiveModel("./model.pkl")
        >>>
        >>> @node(output_name="prediction")
        ... def predict(text: str, model: ExpensiveModel) -> str:
        ...     return model.predict(text)  # __init__ called here on first use

    Caching:
        The wrapper automatically generates a cache key from init arguments.
        For custom caching, implement __cache_key__() on the wrapped class:

        >>> @stateful
        ... class Model:
        ...     def __init__(self, model_path: str):
        ...         self.model = load_model(model_path)
        ...
        ...     def __cache_key__(self) -> str:
        ...         return f"Model(path={self.model_path})"

    Args:
        cls: Class to wrap with lazy initialization

    Returns:
        Wrapper class that delays initialization
    """

    class StatefulWrapper:
        """Lazy initialization wrapper for stateful resources."""

        # Mark as stateful for engine detection
        __hypernode_stateful__ = True

        def __init__(self, *args, **kwargs):
            """Store init arguments without calling wrapped class __init__."""
            self._init_args = args
            self._init_kwargs = kwargs
            self._instance = None  # Lazy initialized
            self._original_class = cls

        def _ensure_initialized(self):
            """Initialize instance if not already done."""
            if self._instance is None:
                self._instance = self._original_class(
                    *self._init_args, **self._init_kwargs
                )

        def __call__(self, *args, **kwargs):
            """Forward calls to wrapped instance."""
            self._ensure_initialized()
            if hasattr(self._instance, "__call__"):
                return self._instance(*args, **kwargs)
            raise TypeError(f"{self._original_class.__name__} object is not callable")

        def __getattr__(self, name: str):
            """Forward attribute access to wrapped instance."""
            if name.startswith("_"):
                # Private attributes on wrapper itself
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )

            # Lazily initialize and forward
            self._ensure_initialized()
            return getattr(self._instance, name)

        def __cache_key__(self) -> str:
            """Generate cache key from init arguments.

            If the wrapped class has __cache_key__(), use that after initialization.
            Otherwise, hash the class name and init arguments.
            """
            # Check if wrapped class has custom __cache_key__
            if hasattr(self._original_class, "__cache_key__"):
                self._ensure_initialized()
                return self._instance.__cache_key__()

            # Default: hash class name and init args
            args_str = str(self._init_args)
            kwargs_str = str(sorted(self._init_kwargs.items()))
            combined = f"{cls.__name__}:{args_str}:{kwargs_str}"
            return hashlib.sha256(combined.encode()).hexdigest()

        def __getstate__(self):
            """Pickle only init arguments, not the instance."""
            return {
                "_init_args": self._init_args,
                "_init_kwargs": self._init_kwargs,
                "_original_class": self._original_class,
                "_instance": None,  # Don't pickle the instance!
            }

        def __setstate__(self, state):
            """Restore from pickle."""
            self._init_args = state["_init_args"]
            self._init_kwargs = state["_init_kwargs"]
            self._original_class = state["_original_class"]
            self._instance = None  # Will be lazily initialized on first use

        def __repr__(self):
            """String representation."""
            if self._instance is None:
                return f"<{cls.__name__} (not initialized)>"
            return repr(self._instance)

    # Preserve class name and module for better debugging
    StatefulWrapper.__name__ = cls.__name__
    StatefulWrapper.__qualname__ = cls.__qualname__
    StatefulWrapper.__module__ = cls.__module__

    return StatefulWrapper
