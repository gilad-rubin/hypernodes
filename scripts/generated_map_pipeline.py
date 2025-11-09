"""
Generated Daft code - Exact translation from HyperNodes pipeline.

This code produces identical results to the HyperNodes pipeline execution.
You can run this file directly to verify the translation.

======================================================================
PERFORMANCE ANALYSIS
======================================================================

⚠️  DETECTED 1 NESTED MAP OPERATIONS

Each map operation creates an explode → process → groupby cycle,
which forces data materialization. This can be 1x slower than optimal.

OPTIMIZATION STRATEGIES:

1. BATCH UDFs: Use @daft.func.batch or @daft.method.batch
   - Processes entire Series at once (10-100x faster for ML models)
   - Eliminates explode/groupby overhead

2. RESTRUCTURE PIPELINE: Reduce nesting
   - Batch encode all passages/queries upfront
   - Use vectorized operations where possible

3. STATEFUL UDFs: Ensure using @daft.cls correctly
   - Initialize expensive objects ONCE per worker
   - Don't pass via daft.lit() - use directly!

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
def sum_all_2(doubled_numbers: Any):
    """Sum all doubled numbers."""
    return sum(doubled_numbers)


# ==================== Pipeline Execution ====================

# Create DataFrame with input data
df = daft.from_pydict({
    "numbers": [[1, 2, 3, 4, 5]],
})

# Map over: numbers
df = df.with_column("__daft_row_id_1__", daft.lit(0))
df = df.with_column("__original_numbers__", df["numbers"])
df = df.explode(daft.col("numbers"))
df = df.with_column("x", df["numbers"])
df = df.with_column("doubled", double_1(df["x"]))
df = df.groupby(daft.col("__daft_row_id_1__")).agg(daft.col("doubled_numbers").list_agg().alias("doubled_numbers"))
# Remove temporary row_id column
df = df.with_column("total", sum_all_2(df["doubled_numbers"]))

# Select output columns
df = df.select(df["doubled_numbers"], df["total"])

# Collect results
result = df.collect()
print(result.to_pydict())