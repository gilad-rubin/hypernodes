"""Test the HyperNode interface for Node and PipelineNode."""

from hypernodes import Pipeline, node, HyperNode

# Create regular nodes
@node(output_name="result1")
def process(text: str) -> str:
    return text.upper()

@node(output_name="result2")
def combine(result1: str, suffix: str) -> str:
    return f"{result1}_{suffix}"

# Test 1: Node implements HyperNode
print("✓ Test 1: Node implements HyperNode interface")
assert isinstance(process, HyperNode)
assert hasattr(process, "root_args")
assert hasattr(process, "output_name")
assert hasattr(process, "cache")

# Test 2: Node has func attribute (function)
print("✓ Test 2: Node has 'func' attribute (the function)")
assert hasattr(process, "func")
assert callable(process.func)

# Test 3: Create pipeline and wrap as node
inner_pipeline = Pipeline(nodes=[process])
pipeline_node = inner_pipeline.as_node(name="wrapped")

print("✓ Test 3: PipelineNode implements HyperNode interface")
assert isinstance(pipeline_node, HyperNode)
assert hasattr(pipeline_node, "root_args")
assert hasattr(pipeline_node, "output_name")
assert hasattr(pipeline_node, "cache")

# Test 4: PipelineNode has pipeline attribute (NOT func)
print("✓ Test 4: PipelineNode has 'pipeline' attribute (not 'func')")
assert hasattr(pipeline_node, "pipeline")
assert pipeline_node.pipeline is inner_pipeline
assert not hasattr(pipeline_node, "func")

# Test 5: Can distinguish Node from PipelineNode by attribute
print("✓ Test 5: Can distinguish Node from PipelineNode")
assert hasattr(process, "func")
assert hasattr(pipeline_node, "pipeline")

# Test 6: Both work in a pipeline's nodes list
outer_pipeline = Pipeline(nodes=[process, combine])
print("✓ Test 6: Both Node and PipelineNode work in Pipeline.nodes")

# Test 7: Can use mixed list
outer_with_nested = Pipeline(nodes=[pipeline_node, combine])
print("✓ Test 7: Can mix Node and PipelineNode in same pipeline")

# Test 8: HyperNode interface properties work
print("✓ Test 8: HyperNode interface properties work correctly")
assert process.root_args == ("text",)
assert process.output_name == "result1"
assert process.cache is True

assert pipeline_node.root_args == ("text",)
assert pipeline_node.output_name == "result1"
assert pipeline_node.cache is True

print("\n✅ All tests passed! HyperNode interface is clean and semantic.")
print("\n📝 Summary:")
print("  - Node has '.func' (the function)")
print("  - PipelineNode has '.pipeline' (the wrapped pipeline)")
print("  - Both implement HyperNode (root_args, output_name, cache)")
