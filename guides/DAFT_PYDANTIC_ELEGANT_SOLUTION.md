# Elegant Pydantic Support with DaftBackend

## The Problem

Our initial solution converted Pydantic models to dicts, which works but loses type safety and elegance.

## The Elegant Solution: Use Daft's Native Pydantic Support

Daft natively supports Pydantic models through PyArrow schemas! We just need to follow their patterns.

### Key Pattern from Daft's Tutorial

From the [Daft PDF Tutorial](https://www.getdaft.io/projects/docs/en/stable/user_guide/tutorials/document-processing.html):

```python
@udf(return_dtype=daft_pyarrow_datatype(ParsedPdf))
class LoadDirectAndParsePdf:
    def __call__(self, urls: Series, pdf_bytes: Series) -> Series:
        return Series.from_pylist(
            # NOTE: it is **vital** to call .model_dump() on each Pydantic class.
            [self.handle(u, p).model_dump() for u, p in zip(urls, pdf_bytes)]
        )
```

### Required Utilities

First, we need the PyArrow-to-Daft conversion utilities:

```python
from typing import Any, get_args, get_origin
import pyarrow
from pydantic import BaseModel
from pydantic_to_pyarrow import get_pyarrow_schema
import daft


def daft_pyarrow_datatype(f_type: type[Any]) -> daft.DataType:
    """Convert a Python type (including Pydantic) to Daft DataType."""
    return daft.DataType.from_arrow_type(pyarrow_datatype(f_type))


def pyarrow_datatype(f_type: type[Any]) -> pyarrow.DataType:
    """Convert Python types to PyArrow types."""
    # Handle Optional types
    if get_origin(f_type) is Union:
        targs = get_args(f_type)
        if len(targs) == 2:
            if targs[0] is type(None) and targs[1] is not type(None):
                refined_inner = targs[1]
            elif targs[0] is not type(None) and targs[1] is type(None):
                refined_inner = targs[0]
            else:
                raise TypeError(f"Cannot convert union {f_type} to PyArrow!")
            inner_type = pyarrow_datatype(refined_inner)
        else:
            raise TypeError(f"Cannot convert general union {f_type} to PyArrow!")
    
    # Handle list types
    elif get_origin(f_type) is list:
        targs = get_args(f_type)
        if len(targs) != 1:
            raise TypeError(f"Expected list with one type, got {targs}")
        element_type = targs[0]
        inner_type = pyarrow.list_(pyarrow_datatype(element_type))
    
    # Handle Pydantic models - this is the key!
    elif issubclass(f_type, BaseModel):
        schema = get_pyarrow_schema(f_type)
        inner_type = pyarrow.struct([(f, schema.field(f).type) for f in schema.names])
    
    # Handle primitive types
    elif issubclass(f_type, str):
        inner_type = pyarrow.string()
    elif issubclass(f_type, int):
        inner_type = pyarrow.int64()
    elif issubclass(f_type, float):
        inner_type = pyarrow.float64()
    elif issubclass(f_type, bool):
        inner_type = pyarrow.bool_()
    else:
        raise TypeError(f"Cannot handle type {f_type}")
    
    return inner_type
```

### Applying to Our Retrieval Pipeline

Here's how to elegantly handle encoding with full Pydantic support:

```python
from hypernodes import node
from pydantic import BaseModel
from typing import Any

class EncodedPassage(BaseModel):
    uuid: str
    text: str
    embedding: Any
    model_config = {"frozen": True, "arbitrary_types_allowed": True}


# Option 1: Simple node (for non-Daft backends)
@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
    """Standard implementation - returns Pydantic model."""
    embedding = encoder.encode(passage.text, is_query=False)
    return EncodedPassage(
        uuid=passage.uuid,
        text=passage.text,
        embedding=embedding
    )


# Option 2: Daft-compatible wrapper
# If we need to make it Daft-compatible, we can use DaftBackend's
# built-in Pydantic handling by ensuring the return type hint is correct.
# DaftBackend should automatically convert Pydantic to PyArrow structs.
```

### The Key: Let DaftBackend Handle It

The real issue is that `DaftBackend._convert_node_to_daft()` needs to:
1. Recognize Pydantic return types
2. Use `daft_pyarrow_datatype()` to get the proper Arrow schema
3. Call `.model_dump()` when serializing

Let me check if DaftBackend already does this...

Actually, looking at the DaftBackend code, it tries to infer types but doesn't have the Pydantic-to-PyArrow conversion utilities!

## The Real Fix: Enhance DaftBackend

The elegant solution is to enhance `DaftBackend._infer_return_dtype()` to support Pydantic models:

```python
def _infer_return_dtype(self, return_type: Any) -> Optional["daft.DataType"]:
    """Infer an appropriate Daft DataType for a node return annotation."""
    import daft
    
    # ... existing code ...
    
    # NEW: Handle Pydantic models!
    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        try:
            from pydantic_to_pyarrow import get_pyarrow_schema
            schema = get_pyarrow_schema(return_type)
            arrow_type = pyarrow.struct([(f, schema.field(f).type) for f in schema.names])
            return daft.DataType.from_arrow_type(arrow_type)
        except ImportError:
            # Fallback to Python object if pydantic_to_pyarrow not available
            return daft.DataType.python()
    
    # ... rest of existing code ...
```

And ensure nodes with Pydantic return types call `.model_dump()`:

```python
def _convert_node_to_daft(self, node: Any, df: "daft.DataFrame", available: Set[str]):
    # ... existing code to get func, params, etc. ...
    
    # Check if return type is Pydantic
    from typing import get_type_hints
    type_hints = get_type_hints(func)
    return_type = type_hints.get("return", None)
    is_pydantic = (
        return_type is not None 
        and isinstance(return_type, type) 
        and issubclass(return_type, BaseModel)
    )
    
    if is_pydantic:
        # Wrap the function to call .model_dump()
        original_func = func
        def wrapped_func(*args, **kwargs):
            result = original_func(*args, **kwargs)
            return result.model_dump() if hasattr(result, 'model_dump') else result
        func = wrapped_func
    
    # ... rest of UDF creation ...
```

## Summary

The elegant solution has three parts:

1. **Add PyArrow conversion utilities** to DaftBackend (copy from Daft tutorial)
2. **Enhance `_infer_return_dtype()`** to detect and convert Pydantic models
3. **Auto-wrap Pydantic-returning functions** to call `.model_dump()`

This way:
- ✅ Users can return Pydantic models naturally
- ✅ Type safety is preserved
- ✅ DaftBackend handles serialization automatically
- ✅ No manual dict conversion needed
- ✅ Works with all Pydantic models (including nested ones)

Would you like me to implement this enhancement to DaftBackend?
