"""Test that PipelineNode.func property works correctly."""

from hypernodes import Pipeline, node

# Create a simple pipeline
@node(output_name="result")
def process(text: str) -> str:
    return text.upper()

inner_pipeline = Pipeline(nodes=[process])

# Wrap it as a node
pipeline_node = inner_pipeline.as_node(name="wrapped")

# Test 1: PipelineNode has func attribute
print("✓ Test 1: PipelineNode has 'func' attribute")
assert hasattr(pipeline_node, "func")

# Test 2: func returns the pipeline
print("✓ Test 2: func returns the wrapped pipeline")
assert pipeline_node.func is inner_pipeline

# Test 3: Can distinguish PipelineNode from regular Node
print("✓ Test 3: Can distinguish PipelineNode from Node")
assert isinstance(pipeline_node.func, Pipeline)
assert not isinstance(process.func, Pipeline)

# Test 4: Regular Node still has func as a function
print("✓ Test 4: Regular Node.func is still a function")
assert callable(process.func)
assert hasattr(process.func, "__name__")

print("\n✅ All tests passed!")
