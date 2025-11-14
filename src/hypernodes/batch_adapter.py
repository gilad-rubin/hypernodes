"""
Batch Adapter Pattern - Think Singular, Run Batch

This module provides decorators to define BOTH singular and batch versions of functions,
allowing you to:
1. Think and debug with singular functions (easier to reason about)
2. Automatically use batch versions for performance when running at scale

Example:
    @batch_optimized
    class TextEncoder:
        '''Singular function - easy to understand!'''
        
        @staticmethod
        def singular(text: str, model: Any) -> List[float]:
            return model.encode(text)
        
        @staticmethod  
        def batch(texts: List[str], model: Any) -> List[List[float]]:
            '''Optimized batch version - 10-100x faster!'''
            return model.encode_batch(texts)
    
    # Use the singular version for pipeline construction (easy!)
    @node(output_name="embedding")
    def encode_text(text: str, model: Any) -> List[float]:
        return TextEncoder.singular(text, model)
    
    # DaftEngine automatically uses batch version when available!
"""

from typing import Any, Callable, List, TypeVar, Optional
from functools import wraps
import inspect

T = TypeVar('T')


class BatchAdapter:
    """Container for singular and batch versions of a function.
    
    Attributes:
        singular: Function that processes one item (for readability/debugging)
        batch: Optimized function that processes many items (for performance)
        name: Function name
    """
    
    def __init__(self, singular_func: Callable, batch_func: Callable, name: str = None):
        self.singular = singular_func
        self.batch = batch_func
        self.name = name or singular_func.__name__
        self.__name__ = self.name
        self.__doc__ = singular_func.__doc__
    
    def __call__(self, *args, **kwargs):
        """Default to singular for direct calls (debugging)."""
        return self.singular(*args, **kwargs)
    
    def __repr__(self):
        return f"BatchAdapter({self.name}, singular={self.singular.__name__}, batch={self.batch.__name__})"


def batch_optimized(cls_or_func=None, *, auto_wrap: bool = True):
    """Decorator to create dual singular/batch versions.
    
    Usage Pattern 1: Class-based (recommended for related operations)
    
        @batch_optimized
        class TextEncoder:
            @staticmethod
            def singular(text: str, model) -> List[float]:
                return model.encode(text)
            
            @staticmethod
            def batch(texts: List[str], model) -> List[List[float]]:
                return model.encode_batch(texts)
    
    Usage Pattern 2: Function-based with explicit batch
    
        @batch_optimized
        def encode_text(text: str, model) -> List[float]:
            return model.encode(text)
        
        @encode_text.batch_version
        def encode_text_batch(texts: List[str], model) -> List[List[float]]:
            return model.encode_batch(texts)
    
    Usage Pattern 3: Auto-wrap (batch calls singular in loop)
    
        @batch_optimized(auto_wrap=True)
        def process_item(item: dict) -> dict:
            # Simple singular logic
            return {"result": item["value"] * 2}
        
        # Automatically creates batch version that loops!
    
    Args:
        cls_or_func: Class or function to decorate
        auto_wrap: If True, automatically create batch version by looping (default: True)
    
    Returns:
        BatchAdapter with singular and batch versions
    """
    
    def decorator(cls_or_func):
        # Pattern 1: Class with singular/batch methods
        if inspect.isclass(cls_or_func):
            if not hasattr(cls_or_func, 'singular'):
                raise ValueError(
                    f"{cls_or_func.__name__} must have a 'singular' method.\n"
                    f"Example:\n"
                    f"  @staticmethod\n"
                    f"  def singular(item, ...):\n"
                    f"      return process(item)"
                )
            
            singular_func = cls_or_func.singular
            
            # Check if batch method exists
            if hasattr(cls_or_func, 'batch'):
                batch_func = cls_or_func.batch
            elif auto_wrap:
                # Auto-create batch version by looping
                batch_func = _create_auto_batch(singular_func)
            else:
                raise ValueError(
                    f"{cls_or_func.__name__} must have a 'batch' method or set auto_wrap=True.\n"
                    f"Example:\n"
                    f"  @staticmethod\n"
                    f"  def batch(items: List, ...):\n"
                    f"      return [process(item) for item in items]"
                )
            
            return BatchAdapter(singular_func, batch_func, cls_or_func.__name__)
        
        # Pattern 2/3: Function-based
        else:
            singular_func = cls_or_func
            
            # Create wrapper that allows adding batch version later
            adapter = BatchAdapter(singular_func, None, singular_func.__name__)
            
            # Add method to set batch version
            def set_batch_version(batch_func):
                adapter.batch = batch_func
                return adapter
            
            adapter.batch_version = set_batch_version
            
            # If auto_wrap, create batch version immediately
            if auto_wrap:
                adapter.batch = _create_auto_batch(singular_func)
            
            return adapter
    
    # Handle decorator with or without arguments
    if cls_or_func is None:
        return decorator
    else:
        return decorator(cls_or_func)


def _create_auto_batch(singular_func: Callable) -> Callable:
    """Create a batch version that loops over the singular function.
    
    This is NOT optimal (still sequential) but allows batch API compatibility.
    For real performance, provide a custom batch implementation.
    """
    @wraps(singular_func)
    def auto_batch_wrapper(items: List[Any], *args, **kwargs):
        """Auto-generated batch version (loops over singular)."""
        return [singular_func(item, *args, **kwargs) for item in items]
    
    auto_batch_wrapper.__doc__ = (
        f"Auto-generated batch version of {singular_func.__name__}.\n"
        f"Note: This loops sequentially. For optimal performance, "
        f"provide a custom batch implementation."
    )
    
    return auto_batch_wrapper


# ==================== Helper: Auto-detect and use batch version ====================

def call_with_batch_if_available(func: Any, items: Any, *args, **kwargs):
    """Helper to automatically use batch version if available.
    
    Args:
        func: Function or BatchAdapter
        items: Single item or list of items
        *args, **kwargs: Additional arguments
    
    Returns:
        Results (singular or batch depending on input)
    """
    # Check if it's a BatchAdapter
    if isinstance(func, BatchAdapter):
        # Determine if items is a list
        if isinstance(items, list):
            # Use batch version
            return func.batch(items, *args, **kwargs)
        else:
            # Use singular version
            return func.singular(items, *args, **kwargs)
    else:
        # Regular function - just call it
        if isinstance(items, list):
            # No batch version, loop manually
            return [func(item, *args, **kwargs) for item in items]
        else:
            return func(items, *args, **kwargs)


# ==================== Integration with HyperNodes ====================

def adapt_for_hypernodes(adapter: BatchAdapter, output_name: str):
    """Create a HyperNodes node from a BatchAdapter.
    
    Returns the SINGULAR version (for ease of reasoning) but marks
    it so DaftEngine can use the batch version if available.
    
    Args:
        adapter: BatchAdapter with singular/batch versions
        output_name: Output name for the node
    
    Returns:
        Node that uses singular for construction, batch for execution
    """
    from hypernodes import node
    
    # Create node with singular version (easy to reason about!)
    singular_node = node(output_name=output_name)(adapter.singular)
    
    # Attach batch version as metadata for DaftEngine to discover
    singular_node._batch_version = adapter.batch
    singular_node._is_batch_optimized = True
    
    return singular_node

