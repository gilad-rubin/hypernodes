#!/usr/bin/env python3
"""
Example showing how to use code generation with the retrieval pipeline.

This is a practical example showing how to add code generation to your existing
Hebrew retrieval pipeline (from mafat_hebrew_retrieval/scripts/daft_test.py).
"""

# Add this to your daft_test.py file after line 776 (after defining inputs)
# to generate Daft code from your pipeline:

EXAMPLE_USAGE = '''
# In your daft_test.py file, after defining the pipeline and inputs:

# Original code (lines 793-800):
if daft:
    engine = DaftEngine(collect=False)
    pipeline = pipeline.with_engine(engine)
else:
    engine = HypernodesEngine(map_executor="sequential", node_executor="sequential")
    pipeline = pipeline.with_engine(engine)

# ADD THIS NEW SECTION - Generate Daft code for analysis:
print("\\n" + "=" * 70)
print("GENERATING DAFT CODE FOR ANALYSIS")
print("=" * 70)

# Generate the code
generated_code = pipeline.show_daft_code(inputs=inputs)

# Save to file
output_file = "generated_retrieval_daft.py"
with open(output_file, "w") as f:
    f.write(generated_code)

print(f"✅ Generated Daft code saved to: {output_file}")
print(f"\\nGenerated {len(generated_code.split(chr(10)))} lines of code")
print("\\nYou can now:")
print("  1. Review the generated code to understand the translation")
print("  2. Identify bottlenecks in the Daft implementation")
print("  3. Hand-optimize the code for better performance")
print("  4. Compare HyperNodes vs native Daft execution times")
print("=" * 70 + "\\n")

# Continue with normal execution...
start_time = time()
print(f"Running retrieval pipeline with {num_examples} examples...")
results = pipeline.run(output_name="evaluation_results", inputs=inputs)
# ... rest of code
'''

# Or, create a separate analysis script:

ANALYSIS_SCRIPT = '''
#!/usr/bin/env python3
"""
Analyze the retrieval pipeline Daft translation.
"""

import sys
sys.path.insert(0, '../mafat_hebrew_retrieval')

from scripts.daft_test import pipeline, inputs, encoder

# Generate code
print("Generating Daft code from retrieval pipeline...")
code = pipeline.show_daft_code(inputs=inputs)

# Save
with open("retrieval_pipeline_daft.py", "w") as f:
    f.write(code)

print("\\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\\nGenerated code: retrieval_pipeline_daft.py")
print(f"Lines of code: {len(code.split(chr(10)))}")

# Print some statistics
num_udfs = code.count("@daft.func") + code.count("@daft.cls")
num_with_columns = code.count("df.with_column")
num_explodes = code.count("explode")
num_groupbys = code.count("groupby")

print(f"\\nStatistics:")
print(f"  - UDF definitions: {num_udfs}")
print(f"  - with_column calls: {num_with_columns}")
print(f"  - explode operations: {num_explodes}")
print(f"  - groupby operations: {num_groupbys}")

print(f"\\nNext steps:")
print("  1. Open retrieval_pipeline_daft.py")
print("  2. Look for sequential operations that could be parallel")
print("  3. Check if map operations are efficient")
print("  4. Compare with hand-written Daft code")
print("=" * 70)
'''


def main():
    print("\n" + "=" * 70)
    print("HOW TO USE CODE GENERATION WITH YOUR RETRIEVAL PIPELINE")
    print("=" * 70)
    
    print("\nOption 1: Add to existing daft_test.py")
    print("-" * 70)
    print(EXAMPLE_USAGE)
    
    print("\n" + "=" * 70)
    print("Option 2: Create separate analysis script")
    print("-" * 70)
    print(ANALYSIS_SCRIPT)
    
    print("\n" + "=" * 70)
    print("✅ RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. Start by generating the code to see what HyperNodes produces

2. Look for these common patterns:
   - Sequential with_column calls (could potentially be parallel)
   - explode -> transform -> groupby (map operations)
   - Multiple passes over the same data
   
3. Compare with native Daft best practices:
   - Use @daft.cls for stateful objects (encoders, models)
   - Batch operations when possible
   - Minimize data movement (explode/groupby)
   
4. Measure before and after:
   - Time the HyperNodes version
   - Time the hand-optimized version
   - Identify what made the difference
   
5. Share findings:
   - If you find inefficiencies, we can improve DaftEngine
   - If hand-optimization helps, document the patterns
   - Help us make the automatic translation better!
""")
    
    print("=" * 70)
    print("Ready to analyze your pipeline!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

