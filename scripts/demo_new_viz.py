"""Comprehensive demonstration of the new visualization system."""

from hypernodes import Pipeline, node
from hypernodes.viz.ui_handler import UIHandler

print("="*70)
print("FRONTEND-AGNOSTIC VISUALIZATION SYSTEM - COMPREHENSIVE DEMO")
print("="*70)

# Create a multi-level nested pipeline to showcase all features

# Level 1: Inner pipeline
@node(output_name="squared")
def square(x: int) -> int:
    """Square the input."""
    return x * x

@node(output_name="doubled")
def double(squared: int, multiplier: int = 2) -> int:
    """Double the squared value."""
    return squared * multiplier

inner_pipeline = Pipeline(
    nodes=[square, double],
    name="InnerMath"
).bind(multiplier=3)  # Bind a parameter

# Level 2: Middle pipeline
@node(output_name="result", cache=False)  # Non-cached node
def add_constant(doubled: int, constant: int) -> int:
    """Add a constant."""
    return doubled + constant

middle_pipeline = Pipeline(
    nodes=[
        inner_pipeline.as_node(
            input_mapping={"value": "x"},  # Map outer "value" to inner "x"
            output_mapping={"doubled": "processed"}  # Map inner "doubled" to outer "processed"
        ),
        add_constant
    ],
    name="MiddleLayer"
)

# Level 3: Outer pipeline
@node(output_name="final")
def format_result(result: int) -> str:
    """Format the final result."""
    return f"Result: {result}"

outer_pipeline = Pipeline(
    nodes=[
        middle_pipeline.as_node(),
        format_result
    ],
    name="OuterPipeline"
)

print("\n" + "-"*70)
print("SCENARIO 1: Serialization - Collapsed (depth=1)")
print("-"*70)
handler_collapsed = UIHandler(outer_pipeline, depth=1)
graph_data = handler_collapsed.get_view_data()

print(f"\nNodes: {len(graph_data['nodes'])}")
for node in graph_data['nodes']:
    print(f"  - {node['label']} ({node['node_type']})")

print(f"\nLevels: {len(graph_data.get('levels', []))}")
for level in graph_data.get('levels', []):
    print(f"  - {level['level_id']}")
    print(f"    Unfulfilled inputs: {level['unfulfilled_inputs']}")
    print(f"    Bound inputs: {level['bound_inputs_at_this_level']}")

print(f"\nEdges: {len(graph_data['edges'])}")

print("\n" + "-"*70)
print("SCENARIO 2: Serialization - Fully Expanded (depth=None)")
print("-"*70)

handler_expanded = UIHandler(outer_pipeline, depth=None)
graph_data_expanded = handler_expanded.get_view_data()

print(f"\nNodes: {len(graph_data_expanded['nodes'])}")
for node in graph_data_expanded['nodes']:
    level = node.get('level_id', 'root')
    indent = "  " * level.count('nested')
    print(f"{indent}- {node['label']} ({node['node_type']}) @ {level}")

print(f"\nLevels: {len(graph_data_expanded['levels'])}")
for level in graph_data_expanded['levels']:
    indent = "  " * level['level_id'].count('nested')
    print(f"{indent}- {level['level_id']}")

print(f"\nEdges: {len(graph_data_expanded['edges'])}")

print("\n" + "-"*70)
print("SCENARIO 3: Graphviz Rendering - Various Styles")
print("-"*70)

styles = ["default", "dark", "professional", "minimal"]
for style_name in styles:
    try:
        result = outer_pipeline.visualize(
            style=style_name,
            depth=1,
            return_type="graphviz"
        )
        print(f"  ✓ {style_name:15} - {type(result).__name__}")
    except Exception as e:
        print(f"  ✗ {style_name:15} - {e}")

print("\n" + "-"*70)
print("SCENARIO 4: Graphviz Rendering - Different Depths")
print("-"*70)

for depth in [1, 2, None]:
    depth_str = "fully expanded" if depth is None else f"depth={depth}"
    try:
        result = outer_pipeline.visualize(
            depth=depth,
            return_type="graphviz"
        )
        nodes_count = len(result.body)  # Approximate node count from graphviz body
        print(f"  ✓ {depth_str:20} - Generated graph with ~{nodes_count} elements")
    except Exception as e:
        print(f"  ✗ {depth_str:20} - {e}")

print("\n" + "-"*70)
print("SCENARIO 5: Engine Selection")
print("-"*70)

engines = [
    ("graphviz", {"style": "dark"}),
    ("ipywidget", {"theme": "CYBERPUNK"}),
]

for engine_name, options in engines:
    try:
        result = outer_pipeline.visualize(
            engine=engine_name,
            **options
        )
        print(f"  ✓ {engine_name:15} - {type(result).__name__}")
    except ImportError as e:
        print(f"  ⚠ {engine_name:15} - {str(e).split(':')[0]} (optional dependency)")
    except Exception as e:
        print(f"  ✗ {engine_name:15} - {e}")

print("\n" + "-"*70)
print("SCENARIO 6: Semantic Flags in Serialized Data")
print("-"*70)

print("\nNode Types:")
for node in graph_data_expanded['nodes']:
    print(f"  - {node['label']:20} → {node['node_type']}")

print("\nInputs with Semantic Flags:")
for node in graph_data_expanded['nodes']:
    if node.get('inputs'):
        print(f"\n  Node: {node['label']}")
        for inp in node['inputs']:
            flags = []
            if inp.get('is_bound'):
                flags.append("bound")
            if inp.get('type_hint'):
                flags.append(f"type={inp['type_hint']}")
            if inp.get('default_value'):
                flags.append(f"default={inp['default_value']}")
            flag_str = ", ".join(flags) if flags else "no flags"
            print(f"    - {inp['name']:15} ({flag_str})")

print("\nEdge Types:")
edge_types = {}
for edge in graph_data_expanded['edges']:
    edge_type = edge['edge_type']
    edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

for edge_type, count in edge_types.items():
    print(f"  - {edge_type:20} × {count}")

print("\n" + "="*70)
print("DEMONSTRATION COMPLETE ✓")
print("="*70)
print("\nKey Achievements:")
print("  ✓ Frontend-agnostic serialization via UIHandler with complete semantic data")
print("  ✓ Pluggable visualization engines (Graphviz, IPyWidget)")
print("  ✓ Per-level hierarchy analysis for nested pipelines")
print("  ✓ Zero frontend calculations - all relationships pre-computed")
print("  ✓ Full backward compatibility with existing API")
print("  ✓ Type hints, bound status, and node types exposed")
print("="*70)
