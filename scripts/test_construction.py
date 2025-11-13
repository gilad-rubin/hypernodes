"""Test basic construction of nodes and pipelines after refactor."""

from hypernodes import Pipeline, node
from hypernodes.pipeline_node import PipelineNode

print("=" * 60)
print("TEST 1: Single node construction")
print("=" * 60)


@node(output_name="result")
def add_one(x: int) -> int:
    return x + 1


print(f"✓ Node created: {add_one}")
print(f"  - output_name: {add_one.output_name}")
print(f"  - root_args: {add_one.root_args}")
print(f"  - cache: {add_one.cache}")
print(f"  - code_hash: {add_one.code_hash[:16]}...")

print("\n" + "=" * 60)
print("TEST 2: Pipeline with single node")
print("=" * 60)

try:
    pipeline = Pipeline(nodes=[add_one])
    print(f"✓ Pipeline created: {pipeline}")
    print(f"  - graph.root_args: {pipeline.graph.root_args}")
    print(f"  - graph.available_output_names: {pipeline.graph.available_output_names}")
    print(f"  - graph.execution_order: {pipeline.graph.execution_order}")
except Exception as e:
    print(f"✗ Pipeline construction failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 3: Pipeline with two sequential nodes")
print("=" * 60)


@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2


@node(output_name="final")
def add_one_to_doubled(doubled: int) -> int:
    return doubled + 1


try:
    pipeline2 = Pipeline(nodes=[double, add_one_to_doubled])
    print(f"✓ Pipeline created: {pipeline2}")
    print(f"  - graph.root_args: {pipeline2.graph.root_args}")
    print(f"  - graph.available_output_names: {pipeline2.graph.available_output_names}")
    print(f"  - graph.execution_order: {pipeline2.graph.execution_order}")
except Exception as e:
    print(f"✗ Pipeline construction failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 4: PipelineNode construction")
print("=" * 60)

try:
    inner_pipeline = Pipeline(nodes=[double])
    pipeline_node = PipelineNode(pipeline=inner_pipeline)
    print(f"✓ PipelineNode created: {pipeline_node}")
    print(f"  - root_args: {pipeline_node.root_args}")
    print(f"  - output_name: {pipeline_node.output_name}")
    print(f"  - cache: {pipeline_node.cache}")
    print(f"  - code_hash: {pipeline_node.code_hash[:16]}...")
except Exception as e:
    print(f"✗ PipelineNode construction failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 5: Nested pipeline construction")
print("=" * 60)

try:

    @node(output_name="tripled")
    def triple(doubled: int) -> int:
        return doubled + doubled // 2

    # Create inner pipeline
    inner = Pipeline(nodes=[double])

    # Wrap as PipelineNode
    inner_node = PipelineNode(pipeline=inner)

    # Create outer pipeline that uses the PipelineNode
    outer = Pipeline(nodes=[inner_node, triple])

    print("✓ Nested pipeline created")
    print(f"  - Inner pipeline outputs: {inner.graph.available_output_names}")
    print(f"  - PipelineNode output: {inner_node.output_name}")
    print(f"  - Outer pipeline root_args: {outer.graph.root_args}")
    print(f"  - Outer pipeline outputs: {outer.graph.available_output_names}")
    print(f"  - Outer execution order: {outer.graph.execution_order}")
except Exception as e:
    print(f"✗ Nested pipeline construction failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("All construction tests completed. Check results above.")
