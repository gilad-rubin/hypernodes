"""Typed interfaces for Pipeline inputs and outputs using TypedDict.

This module provides methods to generate TypedDict classes from pipeline
type hints, enabling IDE autocomplete and static type checking.
"""

from typing import Any, Dict, List, Type, TypedDict, get_type_hints


def _extract_type_from_nodes(param_name: str, nodes: List) -> Any:
    """Extract type hint for a parameter by scanning nodes that use it.
    
    Args:
        param_name: Name of the parameter to find type for
        nodes: List of nodes to scan
        
    Returns:
        Type annotation if found, otherwise Any
    """
    for node in nodes:
        if not hasattr(node, 'func'):
            continue
            
        try:
            hints = get_type_hints(node.func)
            if param_name in hints:
                return hints[param_name]
        except Exception:
            # Fallback to raw annotations
            if hasattr(node.func, '__annotations__'):
                if param_name in node.func.__annotations__:
                    return node.func.__annotations__[param_name]
    
    return Any


def _extract_output_type_from_node(node, output_name: str) -> Any:
    """Extract return type hint for a specific output.
    
    Args:
        node: Node that produces the output
        output_name: Name of the output
        
    Returns:
        Type annotation if found, otherwise Any
    """
    if not hasattr(node, 'func'):
        return Any
        
    try:
        hints = get_type_hints(node.func)
        if 'return' in hints:
            return_type = hints['return']
            
            # Handle multiple outputs (tuple return types)
            if isinstance(node.output_name, tuple):
                # For now, we can't easily parse tuple types, so use Any
                # TODO: Parse Tuple[type1, type2, ...] annotations
                return Any
            
            return return_type
    except Exception:
        pass
    
    return Any


def create_input_type(pipeline) -> Type[TypedDict]:
    """Generate a TypedDict for pipeline inputs.
    
    Args:
        pipeline: Pipeline instance
        
    Returns:
        TypedDict class with fields for each root argument
        
    Example:
        >>> InputType = create_input_type(pipeline)
        >>> inputs: InputType = {"x": 5, "y": 10}
        >>> result = pipeline.run(inputs=inputs)
    """
    annotations = {}
    
    for param_name in pipeline.graph.root_args:
        param_type = _extract_type_from_nodes(param_name, pipeline.nodes)
        annotations[param_name] = param_type
    
    # Generate class name from pipeline name
    class_name = f"{pipeline.name or 'Pipeline'}Input"
    
    # Create TypedDict dynamically
    return TypedDict(class_name, annotations)


def create_output_type(pipeline) -> Type[TypedDict]:
    """Generate a TypedDict for pipeline outputs.
    
    Args:
        pipeline: Pipeline instance
        
    Returns:
        TypedDict class with fields for each output
        
    Example:
        >>> OutputType = create_output_type(pipeline)
        >>> result = pipeline.run(inputs={"x": 5})
        >>> typed_result: OutputType = result
        >>> print(typed_result["result"])  # IDE autocomplete!
    """
    annotations = {}
    
    for output_name in pipeline.graph.available_output_names:
        node = pipeline.graph.output_to_node[output_name]
        output_type = _extract_output_type_from_node(node, output_name)
        annotations[output_name] = output_type
    
    # Generate class name from pipeline name
    class_name = f"{pipeline.name or 'Pipeline'}Output"
    
    return TypedDict(class_name, annotations)


def create_input_constructor(pipeline):
    """Create a constructor function for typed inputs with IDE autocomplete.
    
    This generates a function that provides autocomplete for input parameters.
    
    Args:
        pipeline: Pipeline instance
        
    Returns:
        A constructor function that accepts kwargs and returns a dict
        
    Example:
        >>> make_input = create_input_constructor(pipeline)
        >>> inputs = make_input(x=5, y=10)  # ✅ IDE autocompletes x and y!
        >>> result = pipeline.run(inputs=inputs)
    """
    import inspect
    from typing import get_type_hints
    
    # Build parameter list with type hints
    params = []
    annotations_dict = {}
    
    for param_name in pipeline.graph.root_args:
        param_type = _extract_type_from_nodes(param_name, pipeline.nodes)
        annotations_dict[param_name] = param_type
        params.append(f"{param_name}")
    
    # Create function dynamically
    func_name = f"make_{pipeline.name or 'pipeline'}_input"
    param_str = ", ".join(f"{p}" for p in params)
    
    # Create function code
    code = f"""
def {func_name}({param_str}):
    '''Create typed input dict for pipeline.
    
    Args:
{chr(10).join(f'        {p}: Input parameter' for p in params)}
    
    Returns:
        Input dictionary for pipeline.run()
    '''
    return {{{', '.join(f'"{p}": {p}' for p in params)}}}
"""
    
    # Execute to create function
    namespace = {}
    exec(code, namespace)
    func = namespace[func_name]
    
    # Add type annotations manually
    func.__annotations__ = {**annotations_dict, 'return': Dict[str, Any]}
    
    return func

