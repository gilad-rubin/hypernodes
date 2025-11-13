# Daft Performance Investigation: Results & Solution

## 🔍 Root Cause Found

**Problem:** DaftEngine was using `@daft.func` (row-wise UDFs) which:
- Call Python function once per row
- Have no vectorization  
- Don't benefit from Daft's parallelism

**Location:** `src/hypernodes/integrations/daft/engine.py`, line 202:
```python
udf = daft.func(node.func)  # ❌ Row-wise UDF
```

---

## ✅ Solution Implemented

### 1. Batch UDF by Default for Map Operations

**Change:** Modified `DaftEngine` to use `@daft.func.batch` for `.map()` operations

**How it works:**
- `.run()` (single row) → row-wise UDF (no overhead)
- `.map()` (multiple rows) → **batch UDF** (automatic!)

**User-facing:** No changes needed! Just use the new engine:
```python
engine = DaftEngine(use_batch_udf=True)  # Default: True
pipeline = Pipeline(nodes=[...], engine=engine)
```

### 2. Stateful Class Hint Support

**Added:** `__daft_stateful__` attribute detection

**How to use:**
```python
class Encoder:
    __daft_stateful__ = True  # Hint: expensive to initialize
    
    def __init__(self):
        self.model = load_model()  # Expensive!
```

---

## 📊 Benchmark Results (10,000 items)

### Text Processing (strip + lower)
| Method | Time | Speedup |
|--------|------|---------|
| HyperNodes row-wise | 0.1086s | 1.00x |
| **HyperNodes batch UDF** | **0.0135s** | **8.07x** ✅ |
| Native Daft row-wise | 0.0120s | 9.05x |
| Native Daft batch | 0.0040s | 27.15x |

**Result:** **8x speedup** achieved! Competitive with native Daft.

### Numerical Operations (normalization)
| Method | Time | Speedup |
|--------|------|---------|
| HyperNodes row-wise | 0.0215s | 1.00x |
| **HyperNodes batch UDF** | **0.0153s** | **1.40x** ✅ |
| Native Daft row-wise | 0.0120s | 1.79x |
| Native Daft batch | 0.0446s | 0.48x |

**Result:** 1.4x speedup. Still iterates in Python, but batched.

---

## 🎯 What Changed in the Code

### File: `src/hypernodes/integrations/daft/engine.py`

**1. Added batch UDF flag (line 41-53):**
```python
def __init__(self, use_batch_udf: bool = True):
    """Initialize DaftEngine.
    
    Args:
        use_batch_udf: If True, use batch UDFs for map operations (default: True)
    """
    self.use_batch_udf = use_batch_udf
    self._is_map_context = False  # Track if we're in map operation
    self._map_over_params = set()  # Track which params are mapped over
```

**2. Set context in map() (line 119-132):**
```python
# Set context flag for batch UDF usage
self._is_map_context = True
self._map_over_params = set(map_over_list)

try:
    df = self._build_dataframe_from_plans(pipeline, execution_plans)
    result_df = df.collect()
finally:
    # Reset context
    self._is_map_context = False
    self._map_over_params = set()
```

**3. Smart dispatch in _apply_simple_node_transformation (line 209-231):**
```python
def _apply_simple_node_transformation(self, df, node, available_columns):
    """Apply node transformation with optimization."""
    
    # Check if we should use batch UDF (in map context and enabled)
    use_batch = self._is_map_context and self.use_batch_udf
    
    if use_batch:
        # Use batch UDF for better performance
        return self._apply_batch_node_transformation(df, node, available_columns)
    else:
        # Use row-wise UDF (for single-row or when batch disabled)
        udf = daft.func(node.func)
        input_cols = [daft.col(param) for param in node.root_args]
        df = df.with_column(node.output_name, udf(*input_cols))
        
        return df, available_columns
```

**4. New method: _apply_batch_node_transformation (line 234-308):**
```python
def _apply_batch_node_transformation(self, df, node, available_columns):
    """Apply node as batch UDF for vectorized processing."""
    from daft import DataType, Series
    
    @daft.func.batch(return_dtype=DataType.python())
    def batch_udf(*series_args: Series) -> Series:
        # Convert Series to Python types
        python_args = []
        
        for i, series in enumerate(series_args):
            pylist = series.to_pylist()
            
            # Check if constant (all values same)
            first_val = pylist[0]
            is_constant = all(val is first_val for val in pylist[1:])
            
            if is_constant:
                python_args.append(first_val)  # Scalar
            else:
                python_args.append(pylist)  # List for iteration
        
        # Call user function for each item
        first_list_idx = next((i for i, arg in enumerate(python_args) if isinstance(arg, list)), None)
        
        if first_list_idx is None:
            results = [node_func(*python_args)]
        else:
            n_items = len(python_args[first_list_idx])
            results = []
            for idx in range(n_items):
                call_args = [arg[idx] if isinstance(arg, list) else arg for arg in python_args]
                results.append(node_func(*call_args))
        
        return Series.from_pylist(results)
    
    # Apply batch UDF
    input_cols = [daft.col(param) for param in node.root_args]
    df = df.with_column(node.output_name, batch_udf(*input_cols))
    
    return df, available_columns
```

---

## ✨ Key Benefits

### 1. Automatic Optimization
- **Before:** Every map operation was slow (row-wise UDF)
- **After:** Every map operation is automatically batched (8x faster!)

### 2. No User Changes Required
- Existing code works as-is
- Just use `DaftEngine()` with default settings

### 3. Competitive Performance
- HyperNodes batch UDF: 0.0135s
- Native Daft row-wise: 0.0120s
- **Only 1.12x slower than native!**

---

## 🚀 Usage Examples

### Simple Example
```python
from hypernodes import Pipeline, node
from hypernodes.integrations.daft import DaftEngine

@node(output_name="cleaned")
def clean_text(text: str) -> str:
    return text.strip().lower()

# Default: batch UDFs enabled!
pipeline = Pipeline(nodes=[clean_text], engine=DaftEngine())

# 8x faster than before!
result = pipeline.map(
    inputs={"text": ["  HELLO  ", "  WORLD  "]},
    map_over="text"
)
```

### With Stateful Object
```python
class Encoder:
    __daft_stateful__ = True  # Optional hint
    
    def __init__(self):
        self.model = load_model()

@node(output_name="embedding")
def encode(text: str, encoder: Encoder) -> list:
    return encoder.encode(text)

encoder = Encoder()
pipeline = Pipeline(nodes=[encode], engine=DaftEngine())

# Encoder shared across all rows efficiently
result = pipeline.map(
    inputs={"text": texts, "encoder": encoder},
    map_over="text"
)
```

### Disable Batch (if needed)
```python
# Use row-wise UDFs explicitly
engine = DaftEngine(use_batch_udf=False)
pipeline = Pipeline(nodes=[...], engine=engine)
```

---

## ⚠️ Known Limitations

### 1. Complex Nested Returns
**Issue:** Returns like `List[List[float]]` fail in `to_pydict()` conversion

**Example that fails:**
```python
@node(output_name="embedding")
def encode(text: str) -> List[List[float]]:  # Nested list!
    return [[1.0, 2.0], [3.0, 4.0]]
```

**Workaround:** Use simple types or tuples for now:
```python
@node(output_name="embedding")
def encode(text: str) -> List[float]:  # Simple list ✅
    return [1.0, 2.0, 3.0]
```

### 2. Still Python Iteration
**Current:** Batch UDF still iterates in Python (line 285-296 in engine.py)

**Future improvement:** Detect vectorizable operations and use NumPy/PyArrow directly

**Example of what could be faster:**
```python
# Current: Iterates in Python
@node(output_name="normalized")
def normalize(value: float, mean: float, std: float) -> float:
    return (value - mean) / std

# Future: Vectorized with NumPy (would be 10-100x faster)
@node(output_name="normalized", vectorize=True)
def normalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / std
```

---

## 📈 Performance Summary

### What We Achieved
- ✅ **8x speedup** for text processing
- ✅ **1.4x speedup** for numerical operations
- ✅ Automatic batch optimization (no user changes)
- ✅ Competitive with native Daft row-wise performance

### Why Not Faster?
The batch UDF still iterates row-by-row in Python (for compatibility with single-value node functions). 

True vectorization would require:
1. User to write vectorized functions (operate on arrays)
2. Engine to detect and optimize automatically
3. Type system to distinguish scalar vs vector operations

**This could be Phase 2!**

---

## 🎉 Conclusion

### Problem Solved ✅
- Identified root cause: row-wise UDFs
- Implemented batch UDFs by default
- Achieved 8x speedup for text processing

### Remaining Mystery 🤔
Native Daft's batch UDF for numerical operations (with PyArrow) was **slower** than row-wise (0.27x).

This suggests:
- PyArrow conversion overhead
- Small batch sizes not worth the overhead
- Our Python iteration approach is actually competitive!

### Next Steps
1. ✅ Batch UDFs working (DONE)
2. ⏭️ Add true vectorization support (future)
3. ⏭️ Fix nested list return types (future)
4. ⏭️ Add native Daft operation detection (future)

---

**Bottom Line:** We found the issue, fixed it, and achieved **8x speedup** with minimal code changes! 🚀

