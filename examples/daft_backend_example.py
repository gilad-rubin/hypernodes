"""Example demonstrating DaftBackend for HyperNodes.

This example shows how to use the DaftBackend to automatically convert
HyperNodes pipelines into Daft DataFrames with lazy evaluation and
automatic optimization.
"""

from hypernodes import node, Pipeline
from hypernodes.daft_backend import DaftBackend
import time

print("=" * 60)
print("DaftBackend Example - Automatic HyperNodes to Daft Conversion")
print("=" * 60)

# Example 1: Simple Text Processing
print("\n1. Simple Text Processing")
print("-" * 40)

@node(output_name="cleaned")
def clean_text(text: str) -> str:
    """Clean and normalize text."""
    return text.strip().lower()

@node(output_name="tokens")
def tokenize(cleaned: str) -> list:
    """Split text into tokens."""
    return cleaned.split()

@node(output_name="count")
def count_tokens(tokens: list) -> int:
    """Count number of tokens."""
    return len(tokens)

# Create pipeline with DaftBackend
text_pipeline = Pipeline(
    nodes=[clean_text, tokenize, count_tokens],
    backend=DaftBackend(),
    name="text_processing"
)

# Single execution
result = text_pipeline.run(inputs={"text": "  Hello World  "})
print(f"Single execution: {result}")

# Batch execution with map
texts = [
    "  Hello World  ",
    "  Daft is FAST  ",
    "  HyperNodes are modular  ",
    "  Python for data processing  "
]

start = time.time()
results = text_pipeline.map(inputs={"text": texts}, map_over="text")
elapsed = time.time() - start

print(f"\nBatch execution ({len(texts)} items):")
for i, (cleaned, count) in enumerate(zip(results["cleaned"], results["count"])):
    print(f"  '{texts[i]}' -> '{cleaned}' ({count} tokens)")
print(f"Time: {elapsed:.4f}s")


# Example 2: Diamond Dependency Pattern
print("\n\n2. Diamond Dependency Pattern")
print("-" * 40)

@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="tripled")
def triple(x: int) -> int:
    return x * 3

@node(output_name="sum")
def add(doubled: int, tripled: int) -> int:
    return doubled + tripled

diamond_pipeline = Pipeline(
    nodes=[double, triple, add],
    backend=DaftBackend(),
    name="diamond"
)

result = diamond_pipeline.run(inputs={"x": 5})
print(f"Diamond pattern: x=5 -> doubled={result['doubled']}, tripled={result['tripled']}, sum={result['sum']}")

# Map over multiple values
results = diamond_pipeline.map(inputs={"x": [1, 2, 3, 4, 5]}, map_over="x")
print(f"\nBatch results:")
for i, (x, doubled, tripled, sum_val) in enumerate(zip(
    [1, 2, 3, 4, 5],
    results["doubled"],
    results["tripled"],
    results["sum"]
)):
    print(f"  x={x} -> doubled={doubled}, tripled={tripled}, sum={sum_val}")


# Example 3: Nested Pipelines
print("\n\n3. Nested Pipelines")
print("-" * 40)

@node(output_name="preprocessed")
def preprocess(text: str) -> str:
    return text.strip().lower()

@node(output_name="word_count")
def count_words(preprocessed: str) -> int:
    return len(preprocessed.split())

# Inner pipeline for preprocessing
preprocess_pipeline = Pipeline(
    nodes=[preprocess, count_words],
    name="preprocessing"
)

@node(output_name="is_long")
def classify_length(word_count: int) -> bool:
    return word_count > 3

# Outer pipeline that uses inner pipeline
full_pipeline = Pipeline(
    nodes=[preprocess_pipeline, classify_length],
    backend=DaftBackend(),
    name="full_analysis"
)

texts = [
    "  Short  ",
    "  This is a longer sentence  ",
    "  Hi  ",
    "  Another example with many words here  "
]

results = full_pipeline.map(inputs={"text": texts}, map_over="text")
print("Text analysis results:")
for i, (text, preprocessed, word_count, is_long) in enumerate(zip(
    texts,
    results["preprocessed"],
    results["word_count"],
    results["is_long"]
)):
    print(f"  '{text.strip()}' -> {word_count} words, long={is_long}")


# Example 4: Fixed and Varying Parameters
print("\n\n4. Fixed and Varying Parameters")
print("-" * 40)

@node(output_name="scaled")
def scale(value: float, factor: float) -> float:
    return value * factor

@node(output_name="shifted")
def shift(scaled: float, offset: float) -> float:
    return scaled + offset

transform_pipeline = Pipeline(
    nodes=[scale, shift],
    backend=DaftBackend(),
    name="transform"
)

# Map over values with fixed factor and offset
values = [1.0, 2.0, 3.0, 4.0, 5.0]
results = transform_pipeline.map(
    inputs={"value": values, "factor": 2.5, "offset": 10.0},
    map_over="value"
)

print("Transformation results (factor=2.5, offset=10.0):")
for value, scaled, shifted in zip(values, results["scaled"], results["shifted"]):
    print(f"  {value} -> scaled={scaled}, shifted={shifted}")


# Example 5: Selective Output
print("\n\n5. Selective Output")
print("-" * 40)

@node(output_name="step1")
def step_one(x: int) -> int:
    return x * 2

@node(output_name="step2")
def step_two(step1: int) -> int:
    return step1 + 10

@node(output_name="final")
def step_three(step2: int) -> int:
    return step2 ** 2

selective_pipeline = Pipeline(
    nodes=[step_one, step_two, step_three],
    backend=DaftBackend(),
    name="selective"
)

# Get all outputs
result_all = selective_pipeline.run(inputs={"x": 5})
print(f"All outputs: {result_all}")

# Get only final output
result_final = selective_pipeline.run(inputs={"x": 5}, output_name="final")
print(f"Final only: {result_final}")

# Get specific outputs
result_specific = selective_pipeline.run(inputs={"x": 5}, output_name=["step1", "final"])
print(f"Specific outputs: {result_specific}")


# Example 6: Show Execution Plan
print("\n\n6. Execution Plan Visualization")
print("-" * 40)

# Create backend with show_plan=True
daft_backend_with_plan = DaftBackend(show_plan=True)

simple_pipeline = Pipeline(
    nodes=[double, triple, add],
    backend=daft_backend_with_plan,
    name="with_plan"
)

print("Running pipeline with execution plan:")
result = simple_pipeline.run(inputs={"x": 10})
print(f"Result: {result}")


print("\n" + "=" * 60)
print("DaftBackend examples completed!")
print("=" * 60)
