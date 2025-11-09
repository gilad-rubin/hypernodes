"""
Generated Daft code - Exact translation from HyperNodes pipeline.

This code produces identical results to the HyperNodes pipeline execution.
You can run this file directly to verify the translation.

======================================================================
PERFORMANCE ANALYSIS
======================================================================
For detailed recommendations, see:
https://www.getdaft.io/projects/docs/en/stable/user_guide/udfs.html
======================================================================
"""

import daft
from typing import Any

# ==================== UDF Definitions ====================

@daft.func(return_dtype=daft.DataType.int64())
def double_1(x: Any):
    """Double the input."""
    return x * 2

@daft.func(return_dtype=daft.DataType.int64())
def triple_2(x: Any):
    """Triple the input."""
    return x * 3

@daft.func(return_dtype=daft.DataType.int64())
def add_3(doubled: Any, tripled: Any):
    """Add doubled and tripled values."""
    return doubled + tripled

@daft.func(return_dtype=daft.DataType.int64())
def multiply_with_object_4(x: Any, multiplier: Any):
    """Multiply using stateful object."""
    return multiplier.multiply(x)

@daft.func(return_dtype=daft.DataType.int64())
def add_results_5(result: Any, multiplied: Any):
    """Add both results."""
    return result + multiplied


# ==================== Pipeline Execution ====================

# Create DataFrame with input data
df = daft.from_pydict({
    "x": [5],
})

# Add complex/stateful objects as columns
# Note: These objects must be initialized before running this code
df = df.with_column("multiplier", daft.lit(multiplier))

df = df.with_column("doubled", double_1(df["x"]))
df = df.with_column("tripled", triple_2(df["x"]))
df = df.with_column("result", add_3(df["doubled"], df["tripled"]))
df = df.with_column("multiplied", multiply_with_object_4(df["x"], df["multiplier"]))
df = df.with_column("final_result", add_results_5(df["result"], df["multiplied"]))

# Select output columns
df = df.select(df["doubled"], df["tripled"], df["result"], df["multiplied"], df["final_result"])

# Collect results
result = df.collect()
print(result.to_pydict())