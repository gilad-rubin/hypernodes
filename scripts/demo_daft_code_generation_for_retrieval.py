#!/usr/bin/env python3
"""
Demo script showing how to use Daft code generation with the Hebrew retrieval pipeline.

This script demonstrates:
1. How to generate Daft code from a HyperNodes pipeline
2. How to save it for analysis and optimization
3. How to compare the generated code with what a hand-optimized version would look like

Usage:
    # From the mafat_hebrew_retrieval repository, you can:
    # 1. Import your pipeline
    # 2. Call pipeline.show_daft_code(inputs)
    # 3. Analyze and optimize
"""

from hypernodes import Pipeline, node


def demonstrate_code_generation():
    """Show how to use code generation with any pipeline."""
    
    print("\n" + "=" * 70)
    print("DAFT CODE GENERATION - RETRIEVAL PIPELINE DEMO")
    print("=" * 70)
    
    print("""
This demo shows how to use the new code generation feature with your
Hebrew retrieval pipeline (or any HyperNodes pipeline).

Steps:
------

1. Take your existing pipeline (from daft_test.py):
   
   pipeline = Pipeline(
       nodes=[
           load_passages,
           load_queries,
           ...
       ],
       name="hebrew_retrieval"
   )

2. Generate Daft code:

   generated_code = pipeline.show_daft_code(
       inputs={
           "corpus_path": "data/sample_5/corpus.parquet",
           "examples_path": "data/sample_5/test.parquet",
           "model_name": "lightonai/GTE-ModernColBERT-v1",
           "trust_remote_code": True,
           "encoder": encoder,  # Pre-created encoder
           ...
       }
   )

3. Save to file:

   with open("generated_retrieval_pipeline.py", "w") as f:
       f.write(generated_code)

4. Analyze the generated code:
   - See how HyperNodes translates to Daft
   - Identify bottlenecks (e.g., sequential operations that could be parallel)
   - Understand the explode/groupby pattern for map operations
   - See how stateful objects (encoders) are handled

5. Optimize by hand:
   - Remove unnecessary operations
   - Combine operations where possible
   - Use batch operations for better performance
   - Adjust concurrency settings

6. Compare performance:
   - Run both versions (HyperNodes and hand-optimized)
   - Measure execution time differences
   - Understand where the gains come from

Example Output Structure:
-------------------------

The generated code will look like:

```python
import daft
from typing import Any, List, Dict

# UDF definitions with @daft.cls for stateful objects
@daft.cls(use_process=False)
class ColBERTEncoderWrapper:
    def __init__(self, encoder: Any):
        self.encoder = encoder
    
    @daft.method(return_dtype=daft.DataType.python())
    def __call__(self, passage: Any):
        return self.encoder.encode(passage["text"])

encoder_udf = ColBERTEncoderWrapper(encoder=encoder)

# Pipeline operations
df = daft.from_pydict({...})

# Map operations show explode/groupby pattern
df = df.with_column("__daft_row_id_1__", daft.lit(0))
df = df.explode(daft.col("passages"))
df = df.with_column("encoded_passage", encoder_udf(df["passage"]))
df = df.groupby(daft.col("__daft_row_id_1__")).agg(
    daft.col("encoded_passage").list_agg().alias("encoded_passages")
)

# ... more operations ...

df = df.collect()
```

Benefits:
---------

1. **Transparency**: See exactly what's happening under the hood
2. **Learning**: Understand Daft patterns and best practices
3. **Optimization**: Hand-tune for your specific use case
4. **Debugging**: Identify where performance is lost
5. **Comparison**: Benchmark HyperNodes vs native Daft

Next Steps:
-----------

1. Run your pipeline with code_generation_mode:
   
   from hypernodes.engines import DaftEngine
   code_engine = DaftEngine(code_generation_mode=True)
   pipeline_with_code_gen = pipeline.with_engine(code_engine)
   
   # Or use the convenience method:
   code = pipeline.show_daft_code(inputs=your_inputs)

2. Save the generated code to a file

3. Review and optimize the code

4. Compare performance with the original pipeline

5. Share insights about what patterns are slow

6. We can then improve the DaftEngine translation automatically!
""")
    
    print("=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)
    print("\nTo try this with your retrieval pipeline:")
    print("  1. Navigate to your mafat_hebrew_retrieval directory")
    print("  2. Open scripts/daft_test.py")
    print("  3. Add: code = pipeline.show_daft_code(inputs=inputs)")
    print("  4. Save to file and analyze!")
    print("=" * 70 + "\n")


def create_simple_example():
    """Create a simple example to show the feature."""
    
    print("\n" + "=" * 70)
    print("SIMPLE EXAMPLE")
    print("=" * 70)
    
    # Define simple nodes
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_ten(doubled: int) -> int:
        return doubled + 10
    
    # Create pipeline
    pipeline = Pipeline(nodes=[double, add_ten], name="simple")
    
    # Generate code
    print("\nGenerating Daft code...")
    code = pipeline.show_daft_code(inputs={"x": 5})
    
    print("\n" + "-" * 70)
    print("GENERATED DAFT CODE:")
    print("-" * 70)
    print(code)
    print("-" * 70)
    
    print("\n✅ This code can now be:")
    print("  - Analyzed for bottlenecks")
    print("  - Hand-optimized for performance")
    print("  - Compared with native Daft implementations")
    print("  - Used to understand Daft translation patterns")


if __name__ == "__main__":
    demonstrate_code_generation()
    create_simple_example()

