"""Stateful UDF builder for wrapping functions with stateful dependencies.

This module handles the creation of Daft @daft.cls wrappers around functions
that need access to stateful objects (like ML models, tokenizers, etc).
It follows the Single Responsibility Principle by focusing solely on
stateful UDF construction.
"""

from typing import Any, Callable, Dict, List

try:
    import daft
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False
    daft = None  # type: ignore


class StatefulUDFBuilder:
    """Builds Daft UDFs that capture stateful objects (@daft.cls).

    Stateful objects are things like ML models, tokenizers, or any object
    that should be initialized once and reused across rows. Daft's @daft.cls
    decorator creates a class-based UDF where __init__ runs once per worker
    and __call__ is invoked for each row.

    The builder wraps user functions that need access to stateful objects,
    combining stateful parameters (fixed per worker) with dynamic parameters
    (varying per row from the DataFrame).

    Examples:
        >>> class Encoder:
        ...     __daft_hint__ = "@daft.cls"
        ...     def encode(self, text: str) -> list:
        ...         return [0.1, 0.2, 0.3]
        >>>
        >>> def process(text: str, encoder: Encoder) -> list:
        ...     return encoder.encode(text)
        >>>
        >>> builder = StatefulUDFBuilder()
        >>> udf = builder.build(
        ...     func=process,
        ...     stateful_params={"encoder": encoder},
        ...     dynamic_params=["text"]
        ... )
        >>> # udf is now a @daft.cls UDF that can be applied to DataFrame
    """

    def build(
        self,
        func: Callable,
        stateful_params: Dict[str, Any],
        dynamic_params: List[str],
    ) -> Any:
        """Build a @daft.cls wrapper around function with stateful parameters.

        Args:
            func: Function to wrap
            stateful_params: Dict of parameter_name -> value for stateful params
            dynamic_params: List of parameter names that come from DataFrame

        Returns:
            Daft @daft.cls UDF that can be applied to DataFrame columns

        Examples:
            >>> def encode(text: str, model: Model) -> list:
            ...     return model.encode(text)
            >>>
            >>> udf = builder.build(
            ...     func=encode,
            ...     stateful_params={"model": my_model},
            ...     dynamic_params=["text"]
            ... )
            >>> df = df.with_column("embeddings", udf(df["text"]))
        """
        if not DAFT_AVAILABLE:
            raise ImportError("Daft is not available")

        # Extract @daft.cls configuration from stateful objects
        cls_kwargs = self._extract_daft_cls_config(stateful_params)

        # Create the @daft.cls wrapper
        @daft.cls(**cls_kwargs)
        class StatefulUDF:
            """Wrapper that captures stateful objects in __init__."""

            def __init__(self, stateful_values: Dict[str, Any], dynamic_param_names: List[str]):
                """Initialize with stateful values (runs once per worker).

                Args:
                    stateful_values: Dict of stateful parameter values
                    dynamic_param_names: Names of dynamic parameters (for mapping args)
                """
                self._stateful = stateful_values
                self._dynamic_params = dynamic_param_names

            @daft.method(return_dtype=daft.DataType.python())
            def __call__(self, *args):
                """Execute function with combined stateful + dynamic params.

                Args:
                    *args: Dynamic parameter values from DataFrame columns

                Returns:
                    Function result
                """
                # Combine stateful and dynamic parameters
                kwargs = dict(self._stateful)

                # Map positional args to parameter names
                for param_name, arg_value in zip(self._dynamic_params, args):
                    kwargs[param_name] = arg_value

                # Call original function
                return func(**kwargs)

        # Return instance of the stateful UDF
        return StatefulUDF(stateful_params, dynamic_params)

    def _extract_daft_cls_config(self, stateful_params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract @daft.cls configuration from stateful objects.

        Looks for special __daft_* attributes on stateful objects to configure
        the @daft.cls decorator. Supported attributes:

        - __daft_use_process__: bool - Run in separate process (for PyTorch/CUDA)
        - __daft_max_concurrency__: int - Max concurrent instances
        - __daft_gpus__: int - Number of GPUs to allocate
        - __daft_memory__: str - Memory to allocate (e.g., "4GB")

        Args:
            stateful_params: Dict of stateful parameter values

        Returns:
            Dict of kwargs for @daft.cls decorator

        Examples:
            >>> class Model:
            ...     __daft_use_process__ = True
            ...     __daft_gpus__ = 1
            >>>
            >>> builder._extract_daft_cls_config({"model": Model()})
            {'use_process': True, 'num_gpus': 1}
        """
        config = {}

        for value in stateful_params.values():
            # Check both instance and class for __daft_* attributes
            for obj in [value, type(value)]:
                # use_process: Run in separate process (PyTorch/CUDA compatibility)
                if hasattr(obj, '__daft_use_process__'):
                    config['use_process'] = obj.__daft_use_process__

                # max_concurrency: Limit concurrent instances
                if hasattr(obj, '__daft_max_concurrency__'):
                    config['max_concurrency'] = obj.__daft_max_concurrency__

                # num_gpus: GPU allocation
                if hasattr(obj, '__daft_gpus__'):
                    config['num_gpus'] = obj.__daft_gpus__

                # memory: Memory allocation
                if hasattr(obj, '__daft_memory__'):
                    config['memory'] = obj.__daft_memory__

        # Default: use_process=False for better compatibility
        # (PyTorch works better with threads in most cases)
        if 'use_process' not in config:
            config['use_process'] = False

        return config

    def is_stateful(self, value: Any) -> bool:
        """Check if a value should be treated as stateful.

        A value is stateful if it has the __daft_hint__ attribute, either
        on itself or on its class.

        Args:
            value: Value to check

        Returns:
            True if value should be treated as stateful

        Examples:
            >>> class Model:
            ...     __daft_hint__ = "@daft.cls"
            >>>
            >>> builder = StatefulUDFBuilder()
            >>> builder.is_stateful(Model())
            True
            >>> builder.is_stateful("hello")
            False
        """
        # Check instance
        if hasattr(value, '__daft_hint__'):
            return True

        # Check class
        if hasattr(type(value), '__daft_hint__'):
            return True

        return False

    def extract_stateful_params(
        self,
        params: List[str],
        available_values: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract stateful parameters from available values.

        Args:
            params: List of parameter names
            available_values: Dict of name -> value for all available inputs

        Returns:
            Dict of stateful parameter name -> value

        Examples:
            >>> class Model:
            ...     __daft_hint__ = "@daft.cls"
            >>>
            >>> builder = StatefulUDFBuilder()
            >>> available = {"text": "hello", "model": Model()}
            >>> stateful = builder.extract_stateful_params(
            ...     params=["text", "model"],
            ...     available_values=available
            ... )
            >>> "model" in stateful
            True
            >>> "text" in stateful
            False
        """
        stateful = {}

        for param in params:
            if param in available_values:
                value = available_values[param]
                if self.is_stateful(value):
                    stateful[param] = value

        return stateful
