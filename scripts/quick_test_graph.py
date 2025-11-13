#!/usr/bin/env python3
"""Quick test of graph builder changes."""

from hypernodes import node, Pipeline

@node(output_name="x")
def make_x(input: int) -> int:
    return input * 2

@node(output_name="y")
def make_y(x: int) -> int:
    return x + 10

pipeline = Pipeline(nodes=[make_x, make_y])

print(f"Root args: {pipeline.root_args}")
print(f"Available outputs: {pipeline.available_output_names}")
print(f"Execution order: {[n.func.__name__ for n in pipeline.execution_order]}")
print(f"Dependencies: {len(pipeline.dependencies)} nodes")

# Test execution
result = pipeline(input=5)
print(f"\nResult: {result}")

