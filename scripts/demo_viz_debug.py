#!/usr/bin/env python
"""Demo script showing how to use the visualization debugging tools.

This script demonstrates the debugging workflow for catching viz issues quickly.

Usage:
    uv run python scripts/demo_viz_debug.py
"""

from hypernodes import Pipeline, node
from hypernodes.viz import UIHandler, diagnose_all_states, simulate_state, verify_state

# =============================================================================
# Create a test pipeline with nested structure
# =============================================================================

@node(output_name="cleaned")
def clean_text(text: str) -> str:
    """Clean and normalize text."""
    return text.strip().lower()


@node(output_name="summary")
def summarize(cleaned: str, max_len: int = 100) -> str:
    """Summarize text to max length."""
    return cleaned[:max_len]


@node(output_name="score")
def evaluate(summary: str, expected: str) -> float:
    """Evaluate summary against expected."""
    return 1.0 if summary == expected else 0.0


# Create nested pipeline
inner_pipeline = Pipeline(nodes=[clean_text, summarize], name="text_processor")
inner_node = inner_pipeline.as_node(name="text_processor")
outer_pipeline = Pipeline(nodes=[inner_node, evaluate])


# =============================================================================
# Demo 1: Quick Validation
# =============================================================================

print("=" * 60)
print("Demo 1: Quick Validation")
print("=" * 60)

handler = UIHandler(outer_pipeline, depth=2)
errors = handler.validate_graph()

if errors:
    print("❌ Validation errors found:")
    for err in errors:
        print(f"   - {err}")
else:
    print("✅ Graph validation passed - no errors!")

print()


# =============================================================================
# Demo 2: Debug Dump (Full State Inspection)
# =============================================================================

print("=" * 60)
print("Demo 2: Debug Dump (Full State Inspection)")
print("=" * 60)

dump = handler.debug_dump()

print(f"Stats: {dump['stats']['total_nodes']} nodes, {dump['stats']['total_edges']} edges")
print(f"Node types: {dump['stats']['node_types']}")
print(f"Expanded nodes: {dump['state']['expanded_nodes']}")
print()

print("Input type hints:")
for name, hint in dump['metadata']['input_type_hints'].items():
    print(f"  {name}: {hint}")

print()
print("Output type hints:")
for name, hint in dump['metadata']['output_type_hints'].items():
    print(f"  {name}: {hint}")

print()


# =============================================================================
# Demo 3: Trace Specific Node
# =============================================================================

print("=" * 60)
print("Demo 3: Trace Specific Node")
print("=" * 60)

# Trace an input node
trace = handler.trace_node("text")
print("Tracing 'text' node:")
print(f"  Status: {trace['status']}")
print(f"  Type: {trace.get('node_type', 'N/A')}")
if 'data_info' in trace:
    print(f"  Is input: {trace['data_info']['is_input']}")
    print(f"  Type hint: {trace['data_info']['type_hint']}")

print()

# Trace a function node
trace = handler.trace_node("evaluate")
print("Tracing 'evaluate' node:")
print(f"  Status: {trace['status']}")
print(f"  Type: {trace.get('node_type', 'N/A')}")
if 'outputs' in trace:
    print(f"  Outputs: {trace['outputs']}")
if 'incoming_edges' in trace:
    print(f"  Incoming edges: {[e['from'] for e in trace['incoming_edges']]}")

print()


# =============================================================================
# Demo 4: Trace Edge (Debug Missing Edges)
# =============================================================================

print("=" * 60)
print("Demo 4: Trace Edge (Debug Missing Edges)")
print("=" * 60)

# Trace an existing edge
trace = handler.trace_edge("expected", "evaluate")
print("Tracing 'expected' → 'evaluate':")
print(f"  Edge found: {trace['edge_found']}")

# Trace a potentially missing edge
trace = handler.trace_edge("text_processor", "evaluate")
print()
print("Tracing 'text_processor' → 'evaluate':")
print(f"  Edge found: {trace['edge_found']}")
if not trace['edge_found'] and 'analysis' in trace:
    print(f"  Edges from source: {trace['analysis'].get('edges_from_source', [])}")
    print(f"  Edges to target: {trace['analysis'].get('edges_to_target', [])}")

print()


# =============================================================================
# Demo 5: Simulate State (Test JS Transformations in Python)
# =============================================================================

print("=" * 60)
print("Demo 5: Simulate State (Test JS Transformations)")
print("=" * 60)

graph = handler.get_visualization_data(traverse_collapsed=True)

# Test collapsed + combined mode
result = simulate_state(
    graph,
    expansion_state={"text_processor": False},
    separate_outputs=False,
)

visible = [n["id"] for n in result["nodes"] if not n.get("hidden")]
edges = [(e["source"], e["target"]) for e in result["edges"]]

print("Collapsed + Combined mode:")
print(f"  Visible nodes: {visible}")
print(f"  Edges: {edges}")

# Verify expectations
verification = verify_state(
    result,
    expected_edges=[("text_processor", "evaluate")],
    forbidden_edges=[("text_processor__clean_text", "evaluate")],  # Internal nodes hidden
)

print()
print(f"  Verification passed: {verification['passed']}")
if not verification['passed']:
    print(f"  Failures: {verification['failures']}")

print()


# =============================================================================
# Demo 6: Diagnose All States (AI Agent One-Shot)
# =============================================================================

print("=" * 60)
print("Demo 6: Diagnose All States (AI Agent One-Shot)")
print("=" * 60)

results = diagnose_all_states(graph)

print("State combinations analysis:")
for key, state in results.items():
    orphans = state['orphan_edges']
    status = "✅" if not orphans else f"❌ ({len(orphans)} orphan edges)"
    print(f"  {key}: {state['visible_node_count']} visible, {state['edge_count']} edges {status}")

print()


# =============================================================================
# Demo 7: Find All Issues (Comprehensive Diagnostic)
# =============================================================================

print("=" * 60)
print("Demo 7: Find All Issues (Comprehensive Diagnostic)")
print("=" * 60)

issues = handler.find_issues()

if issues:
    print("Issues found:")
    for category, items in issues.items():
        print(f"  {category}:")
        for item in items:
            print(f"    - {item}")
else:
    print("✅ No issues found!")

print()
print("=" * 60)
print("Debug demo complete!")
print("=" * 60)
