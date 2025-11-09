# Using Code Generation with Your Hebrew Retrieval Pipeline

## ✅ FIXED: Code Generation Now Works Properly!

The issues you encountered have been resolved. The generated Daft code now includes:

1. **Complete DataFrame creation** with actual input data
2. **Full map operation sequences** (explode → inner ops → groupby)
3. **Output collection** (select → collect → print)
4. **Proper handling of stateful objects** (like your encoder)

## How to Use with Your Retrieval Pipeline

### Quick Usage

Add this to your script after defining the pipeline:

```python
# Generate Daft code
code = pipeline.show_daft_code(
    inputs=inputs,
    output_name="evaluation_results"
)

# Print or save it
print(code)

# Save to file for analysis
with open("generated_retrieval_daft.py", "w") as f:
    f.write(code)
```

### What You'll Get

The generated code will show:

1. **All UDF definitions** - your @daft.func and @daft.cls wrappers
2. **DataFrame creation** - `daft.from_pydict()` with your actual inputs
3. **All pipeline operations** - the exact sequence of transformations
4. **Map operations** - how `.map()` translates to Daft's explode/groupby
5. **Stateful object handling** - how encoder/reranker are passed through
6. **Output collection** - how results are selected and collected

### Example Output Structure

```python
"""
Generated Daft code - Exact translation from HyperNodes pipeline.
...
"""

import daft

# ==================== UDF Definitions ====================

@daft.func(return_dtype=List[Python])
def load_passages_1(corpus_path: Any):
    ...

@daft.func(return_dtype=Python)
def encode_passage_2(passage: Any, encoder: Any):
    ...

# More UDFs...

# ==================== Pipeline Execution ====================

# Create DataFrame with input data
df = daft.from_pydict({
    "corpus_path": ['data/sample_10/corpus.parquet'],
    "examples_path": ['data/sample_10/test.parquet'],
    "model_name": ['lightonai/GTE-ModernColBERT-v1'],
    # ... all your inputs
})

# Add stateful objects (need pre-initialization)
df = df.with_column("encoder", daft.lit(encoder))

# Pipeline operations
df = df.with_column("passages", load_passages_1(df["corpus_path"]))

# Map operation for encoding passages
df = df.with_column("__daft_row_id_1__", daft.lit(0))
df = df.with_column("__original_passages__", df["passages"])
df = df.explode(daft.col("passages"))
df = df.with_column("encoded_passage", encode_passage_2(df["passage"], df["encoder"]))
df = df.groupby(daft.col("__daft_row_id_1__")).agg(
    daft.col("encoded_passages").list_agg().alias("encoded_passages")
)

# More operations...

# Select output columns
df = df.select(df["evaluation_results"])

# Collect results
result = df.collect()
print(result.to_pydict())
```

## Analyzing Performance Bottlenecks

With the generated code, you can:

### 1. Compare with Native Daft

Write a native Daft version and compare:
- Are we doing unnecessary operations?
- Are we materializing data too early?
- Can we vectorize operations better?

### 2. Identify Inefficiencies

Look for:
- **Multiple explode/groupby cycles** - each `.map()` adds one
- **Stateful object passing** - encoder being passed through every row
- **Unnecessary columns** - are we keeping columns we don't need?

### 3. Hand-Optimize

Take the generated code and optimize:
- Remove intermediate columns
- Combine operations where possible
- Use Daft-native operations instead of Python UDFs where applicable

### 4. Measure Differences

Run both versions and compare:
```python
import time

# HyperNodes version
start = time.time()
results1 = pipeline.run(inputs=inputs, output_name="evaluation_results")
time1 = time.time() - start

# Hand-optimized Daft version (from generated code)
start = time.time()
results2 = run_optimized_daft_pipeline(inputs)
time2 = time.time() - start

print(f"HyperNodes: {time1:.2f}s")
print(f"Optimized:  {time2:.2f}s")
print(f"Speedup:    {time1/time2:.2f}x")
```

## Known Limitations

1. **Stateful objects require pre-initialization** - The generated code shows where encoder/reranker need to be initialized
2. **Code may not be immediately runnable** - You may need to adjust imports or initialization
3. **Complex objects shown as comments** - Non-serializable objects are noted but not fully generated

## Next Steps

1. **Generate the code** for your full retrieval pipeline
2. **Analyze the operations** - understand what Daft is actually doing
3. **Identify bottlenecks** - where is time being spent?
4. **Compare with native Daft** - write an optimized version
5. **Report back** - what did you find? Where can we improve?

## Example: Finding the Performance Issue

Your concern was: "When I run this code with daft backend I get the same runtime as sequential."

With code generation, you can now:

1. Generate the code: `code = pipeline.show_daft_code(...)`
2. Analyze what operations are being performed
3. Check if Daft is actually parallelizing (look for UDF definitions, are they using @daft.cls properly?)
4. Identify if the bottleneck is:
   - Encoder initialization (being done per-row?)
   - Too many explode/groupby cycles
   - Daft not being given enough parallelism hints
   - Data transfer overhead

Then you can experiment with:
- Different executor strategies
- Batching operations differently
- Using Daft's native operations more
- Adjusting concurrency settings

---

**The tool is ready to use! Try it with your retrieval pipeline and let me know what you discover!**

