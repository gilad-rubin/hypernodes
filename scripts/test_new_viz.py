"""Test script to verify the new visualization system works correctly."""

from hypernodes import Pipeline, node

# Create simple test pipeline
@node(output_name="doubled")
def double(x: int) -> int:
    """Double the input."""
    return x * 2

@node(output_name="result")
def add_ten(doubled: int) -> int:
    """Add ten to the input."""
    return doubled + 10

# Create pipeline
pipeline = Pipeline(nodes=[double, add_ten], name="TestPipeline")

# Test 1: Basic visualization with default engine (graphviz)
print("Test 1: Default graphviz visualization...")
try:
    result = pipeline.visualize(filename="test_output", return_type="graphviz")
    print(f"✓ Success! Result type: {type(result)}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Serialization only
print("\nTest 2: Serialization...")
try:
    from hypernodes.viz.graph_serializer import GraphSerializer
    serializer = GraphSerializer(pipeline)
    graph_data = serializer.serialize(depth=1)
    print(f"✓ Success! Serialized {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
    print(f"  Levels: {[level['level_id'] for level in graph_data['levels']]}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Explicit engine selection
print("\nTest 3: Explicit graphviz engine...")
try:
    from hypernodes.viz.visualization_engines import GraphvizEngine
    engine = GraphvizEngine()
    result = pipeline.visualize(engine=engine, return_type="graphviz")
    print(f"✓ Success! Result type: {type(result)}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Legacy parameters still work
print("\nTest 4: Legacy parameters (backward compatibility)...")
try:
    result = pipeline.visualize(
        orient="LR",
        style="dark",
        show_legend=True,
        return_type="graphviz"
    )
    print(f"✓ Success! Result type: {type(result)}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Nested pipeline
print("\nTest 5: Nested pipeline...")
try:
    @node(output_name="squared")
    def square(x: int) -> int:
        return x * x
    
    inner_pipeline = Pipeline(nodes=[square], name="InnerPipeline")
    
    @node(output_name="final")
    def process(squared: int) -> int:
        return squared + 5
    
    outer_pipeline = Pipeline(
        nodes=[inner_pipeline.as_node(), process],
        name="OuterPipeline"
    )
    
    serializer = GraphSerializer(outer_pipeline)
    graph_data = serializer.serialize(depth=None)  # Fully expanded
    print(f"✓ Success! Serialized {len(graph_data['nodes'])} nodes, {len(graph_data['levels'])} levels")
    print(f"  Levels: {[level['level_id'] for level in graph_data['levels']]}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("All tests completed!")
print("="*50)

