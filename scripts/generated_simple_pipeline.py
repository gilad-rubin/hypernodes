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


# ==================== Pipeline Execution ====================

# Create DataFrame with input data
df = daft.from_pydict({
    "x": [5],
})

df = df.with_column("doubled", double_1(df["x"]))
df = df.with_column("tripled", triple_2(df["x"]))
df = df.with_column("result", add_3(df["doubled"], df["tripled"]))

# Select output columns
df = df.select(df["doubled"], df["tripled"], df["result"])

# Collect results
result = df.collect()
print(result.to_pydict())