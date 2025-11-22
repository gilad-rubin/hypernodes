"""Test script to verify the new visualization system works correctly."""

from hypernodes import Pipeline, node
from hypernodes.viz.ui_handler import UIHandler

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
    handler = UIHandler(pipeline)
    graph_data = handler.get_view_data()
    print(
        f"✓ Success! Serialized {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges"
    )
    print(f"  Levels: {[level['level_id'] for level in graph_data.get('levels', [])]}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Explicit engine selection
print("\nTest 3: Explicit graphviz engine...")
try:
    from hypernodes.viz.graphviz_ui import GraphvizEngine
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
    
    handler = UIHandler(outer_pipeline, depth=None)
    graph_data = handler.get_view_data()
    print(
        f"✓ Success! Serialized {len(graph_data['nodes'])} nodes, "
        f"{len(graph_data.get('levels', []))} levels"
    )
    print(f"  Levels: {[level['level_id'] for level in graph_data.get('levels', [])]}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("All tests completed!")
print("="*50)
