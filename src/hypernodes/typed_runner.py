"""Typed runner builder for fluent pipeline execution with IDE autocomplete."""

from typing import Any, Dict, List, Literal, Optional, Union, get_args


class TypedRunner:
    """Fluent builder for typed pipeline execution with IDE autocomplete.
    
    This provides a builder pattern where inputs are provided as keyword arguments
    with full IDE autocomplete support.
    
    Example:
        >>> result = pipeline.with_inputs(x=5, y=10).run()
        >>> results = pipeline.with_map_inputs(x=[1,2,3], y=10).map_over("x").run()
    """
    
    def __init__(self, pipeline, is_map_mode: bool = False, **inputs):
        """Initialize with pipeline and input values.
        
        Args:
            pipeline: Pipeline instance
            is_map_mode: Whether this is for map execution (allows lists)
            **inputs: Input values (with IDE autocomplete from constructor)
        """
        self._pipeline = pipeline
        self._inputs = inputs
        self._is_map_mode = is_map_mode
        self._map_over: Optional[Union[str, List[str]]] = None
        self._map_mode: Literal["zip", "product"] = "zip"
        self._output_name: Union[str, List[str], None] = None
        
        # Validate inputs if not map mode (no lists allowed for single execution)
        if not is_map_mode:
            self._validate_scalar_inputs()
    
    def _validate_scalar_inputs(self):
        """Validate that inputs match expected types (no lists unless type IS list).
        
        This allows:
        - items: List[int] with value [1,2,3] ✅ (type IS list)
        - x: int with value 5 ✅ (scalar)
        
        This rejects:
        - x: int with value [1,2,3] ❌ (trying to map)
        """
        from typing import get_origin
        
        from .typed_interface import _extract_type_from_nodes
        
        for param_name, param_value in self._inputs.items():
            if not isinstance(param_value, list):
                continue
            
            # Parameter value is a list - check if the type allows it
            param_type = _extract_type_from_nodes(param_name, self._pipeline.nodes)
            
            # Check if the type origin is list (e.g., List[int], list, etc.)
            type_origin = get_origin(param_type)
            
            # If type is not a list type, reject the list value
            if type_origin not in (list, List):
                raise TypeError(
                    f"with_inputs() received a list for parameter '{param_name}', "
                    f"but its type is {param_type}, not List. "
                    f"Use with_map_inputs() to map over lists instead."
                )
    
    def _validate_map_inputs(self, map_over_params: List[str]):
        """Validate map inputs consistency.
        
        Args:
            map_over_params: List of parameter names to map over
            
        Raises:
            TypeError: If mapped params are not lists or non-mapped params are lists
        """
        # Check that mapped params are actually lists
        for param in map_over_params:
            if param in self._inputs and not isinstance(self._inputs[param], list):
                raise TypeError(
                    f"Parameter '{param}' specified in map_over but value is not a list. "
                    f"Got: {type(self._inputs[param]).__name__}"
                )
        
        # Check that non-mapped params are NOT lists (they should be broadcast values)
        non_mapped_lists = [
            k for k in self._inputs.keys()
            if k not in map_over_params and isinstance(self._inputs[k], list)
        ]
        if non_mapped_lists:
            raise TypeError(
                f"Parameters {non_mapped_lists} are lists but not in map_over. "
                f"Either include them in map_over or provide scalar values for broadcasting."
            )
    
    def map(
        self,
        map_over: Union[str, List[str]],
        map_mode: Literal["zip", "product"] = "zip",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute map operation over specified parameters.
        
        Args:
            map_over: Parameter name(s) to map over (must be lists)
            map_mode: "zip" (parallel iteration) or "product" (all combinations)
            **kwargs: Additional engine-specific parameters
            
        Returns:
            List of result dictionaries
            
        Example:
            >>> # Map over single parameter
            >>> results = pipeline.with_map_inputs(x=[1,2,3], y=10).map(map_over="x")
            >>>
            >>> # Map over multiple (zip mode)
            >>> results = pipeline.with_map_inputs(x=[1,2,3], y=[10,20,30]).map(
            ...     map_over=["x", "y"]
            ... )
            >>>
            >>> # Map over multiple (product mode)
            >>> results = pipeline.with_map_inputs(x=[1,2], y=[10,20]).map(
            ...     map_over=["x", "y"],
            ...     map_mode="product"
            ... )
        """
        if not self._is_map_mode:
            raise RuntimeError(
                "map() can only be called on with_map_inputs(), not with_inputs(). "
                "Use pipeline.with_map_inputs(...).map(map_over=...)"
            )
        
        # Normalize map_over to list
        map_params_list = [map_over] if isinstance(map_over, str) else map_over
        
        # Validate that mapped params are lists and non-mapped are scalars
        self._validate_map_inputs(map_params_list)
        
        # Execute immediately
        return self._pipeline.map(
            inputs=self._inputs,
            map_over=map_over,
            map_mode=map_mode,
            output_name=self._output_name,
            **kwargs
        )
    
    def with_output_name(self, *output_names: str) -> "TypedRunner":
        """Specify which outputs to compute (optimization).
        
        Args:
            *output_names: Names of outputs to compute
            
        Returns:
            Self for chaining
            
        Example:
            >>> pipeline.with_inputs(x=5).with_output_name("result1").run()
        """
        self._output_name = list(output_names) if len(output_names) > 1 else output_names[0]
        return self
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the pipeline with configured inputs (single execution only).
        
        Args:
            **kwargs: Additional engine-specific parameters
            
        Returns:
            Results dictionary
            
        Raises:
            RuntimeError: If called on with_map_inputs() (use .map() instead)
            
        Example:
            >>> result = pipeline.with_inputs(x=5, y=10).run()
        """
        if self._is_map_mode:
            # Map mode - must call .map() instead
            raise RuntimeError(
                "Cannot call .run() on with_map_inputs(). "
                "Use .map() to execute: pipeline.with_map_inputs(x=[1,2,3], y=10).map(map_over='x')"
            )
        
        # Single execution
        return self._pipeline.run(
            inputs=self._inputs,
            output_name=self._output_name,
            **kwargs
        )
    


def create_typed_runner_constructor(pipeline, is_map_mode: bool = False):
    """Create a constructor that returns TypedRunner with typed inputs.
    
    This generates a function with proper type hints that provides IDE
    autocomplete for input parameters and returns a TypedRunner builder.
    
    Args:
        pipeline: Pipeline instance
        is_map_mode: If True, creates Union[T, List[T]] types for map inputs
        
    Returns:
        Constructor function that accepts kwargs and returns TypedRunner
        
    Example:
        >>> # Single execution (scalar types only)
        >>> with_inputs = create_typed_runner_constructor(pipeline, is_map_mode=False)
        >>> result = with_inputs(x=5, y=10).run()  # ✅ Autocomplete!
        >>>
        >>> # Map execution (Union[T, List[T]] types)
        >>> with_map_inputs = create_typed_runner_constructor(pipeline, is_map_mode=True)
        >>> results = with_map_inputs(x=[1,2,3], y=10).map_over("x").run()
    """
    from .typed_interface import _extract_type_from_nodes
    
    # Build parameter annotations
    annotations_dict = {}
    params = []
    
    for param_name in pipeline.graph.root_args:
        param_type = _extract_type_from_nodes(param_name, pipeline.nodes)
        
        # For map mode, allow both scalar and list types
        if is_map_mode:
            param_type = Union[param_type, List[param_type]]
        
        annotations_dict[param_name] = param_type
        params.append(param_name)
    
    # Create function dynamically
    func_name = "with_map_inputs" if is_map_mode else "with_inputs"
    param_str = ", ".join(params)
    
    # Create function code
    if is_map_mode:
        docstring = '''Create typed runner with map inputs for pipeline.
    
    Args:
{chr(10).join(f'        {p}: Input parameter (scalar for broadcast, list for mapping)' for p in params)}
    
    Returns:
        TypedRunner for map execution
        
    Example:
        >>> results = with_map_inputs(x=[1,2,3], y=10).map_over("x").run()
    '''
    else:
        docstring = '''Create typed runner with inputs for pipeline.
    
    Args:
{chr(10).join(f'        {p}: Input parameter (scalar only, no lists)' for p in params)}
    
    Returns:
        TypedRunner for single execution
        
    Example:
        >>> result = with_inputs(x=5, y=10).run()
    '''
    
    code = f"""
def {func_name}({param_str}):
    '''{docstring}'''
    return TypedRunner(pipeline, is_map_mode={is_map_mode}, {', '.join(f'{p}={p}' for p in params)})
"""
    
    # Execute to create function
    namespace = {"TypedRunner": TypedRunner, "pipeline": pipeline}
    exec(code, namespace)
    func = namespace[func_name]
    
    # Add type annotations
    func.__annotations__ = {**annotations_dict, 'return': TypedRunner}
    
    return func
