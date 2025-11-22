import pytest
from hypernodes import Pipeline, node
from hypernodes.viz.ui_handler import UIHandler

# --- Setup ---

@node(output_name="a_out")
def func_a(x: int) -> int:
    return x + 1

@node(output_name="b_out")
def func_b(y: int) -> int:
    return y * 2

@node(output_name="c_out")
def func_c(z: int) -> int:
    return z - 1

def create_nested_pipeline():
    # Level 3
    # p3 needs 'z', outputs 'c_out'
    p3 = Pipeline(nodes=[func_c], name="p3")
    
    # Level 2
    # p2 needs 'y'
    # func_b produces 'b_out' from 'y'
    # p3 needs 'z'. We map 'z' to 'b_out' so p3 consumes func_b's output
    n_p3 = p3.as_node(input_mapping={"b_out": "z"})
    p2 = Pipeline(nodes=[func_b, n_p3], name="p2")
    
    # Level 1 (Root)
    # p1 needs 'x'
    # func_a produces 'a_out' from 'x'
    # p2 needs 'y'. We map 'y' to 'a_out' so p2 consumes func_a's output
    n_p2 = p2.as_node(input_mapping={"a_out": "y"})
    p1 = Pipeline(nodes=[func_a, n_p2], name="p1")
    
    return p1

# --- Tests ---

def test_ui_handler_initialization():
    pipeline = create_nested_pipeline()
    handler = UIHandler(pipeline)
    
    # Verify full graph is extracted
    assert handler.full_graph is not None
    assert "nodes" in handler.full_graph
    assert "edges" in handler.full_graph
    assert "levels" in handler.full_graph
    
    # Check that we have nodes from all levels
    # We expect: func_a, p2_node, func_b, p3_node, func_c
    nodes = handler.full_graph["nodes"]
    labels = [n.get("label") for n in nodes]
    
    assert "func_a" in labels
    assert "p2" in labels
    assert "func_b" in labels
    assert "p3" in labels
    assert "func_c" in labels

def test_ui_handler_depth_1():
    pipeline = create_nested_pipeline()
    handler = UIHandler(pipeline)
    handler.set_initial_depth(1)
    
    view = handler.get_view_data()
    visible_nodes = view["nodes"]
    
    # At depth 1, we should see:
    # - func_a (root)
    # - p2 (root, collapsed)
    # We should NOT see:
    # - func_b (inside p2)
    # - p3 (inside p2)
    # - func_c (inside p3)
    
    labels = [n.get("label") for n in visible_nodes]
    assert "func_a" in labels
    assert "p2" in labels
    assert "func_b" not in labels
    assert "p3" not in labels
    
    # Verify p2 is NOT expanded
    p2_node = next(n for n in visible_nodes if n.get("label") == "p2")
    assert not p2_node.get("is_expanded")

def test_ui_handler_depth_2():
    pipeline = create_nested_pipeline()
    handler = UIHandler(pipeline)
    handler.set_initial_depth(2)
    
    view = handler.get_view_data()
    visible_nodes = view["nodes"]
    
    # At depth 2, we should see:
    # - func_a (root)
    # - p2 (root, expanded)
    # - func_b (inside p2)
    # - p3 (inside p2, collapsed)
    # We should NOT see:
    # - func_c (inside p3)
    
    labels = [n.get("label") for n in visible_nodes]
    assert "func_a" in labels
    assert "p2" in labels
    assert "func_b" in labels
    assert "p3" in labels
    assert "func_c" not in labels
    
    # Verify p2 IS expanded
    p2_node = next(n for n in visible_nodes if n.get("label") == "p2")
    assert p2_node.get("is_expanded")
    
    # Verify p3 is NOT expanded
    p3_node = next(n for n in visible_nodes if n.get("label") == "p3")
    assert not p3_node.get("is_expanded")

def test_ui_handler_interaction():
    pipeline = create_nested_pipeline()
    handler = UIHandler(pipeline)
    handler.set_initial_depth(1)
    
    # Initial state: p2 collapsed
    view = handler.get_view_data()
    p2_node = next(n for n in view["nodes"] if n.get("label") == "p2")
    p2_id = p2_node["id"]
    assert not p2_node.get("is_expanded")
    
    # Expand p2
    handler.expand_node(p2_id)
    
    # Check view again
    view = handler.get_view_data()
    visible_nodes = view["nodes"]
    labels = [n.get("label") for n in visible_nodes]
    
    # Should now see contents of p2
    assert "func_b" in labels
    assert "p3" in labels
    
    # p2 should be marked expanded
    p2_node_new = next(n for n in visible_nodes if n["id"] == p2_id)
    assert p2_node_new.get("is_expanded")
    
    # Collapse p2
    handler.collapse_node(p2_id)
    
    # Check view again
    view = handler.get_view_data()
    visible_nodes = view["nodes"]
    labels = [n.get("label") for n in visible_nodes]
    
    # Should NOT see contents of p2
    assert "func_b" not in labels
    assert "p3" not in labels

def test_ui_handler_recursive_collapse():
    """Test that collapsing a parent recursively collapses children."""
    pipeline = create_nested_pipeline()
    handler = UIHandler(pipeline)
    
    # Start with everything expanded (Depth 3)
    handler.set_initial_depth(3)
    
    # Verify initial state: p2 and p3 expanded
    full_graph = handler.get_full_graph_with_state()
    p2_node = next(n for n in full_graph["nodes"] if n.get("label") == "p2")
    p3_node = next(n for n in full_graph["nodes"] if n.get("label") == "p3")
    assert p2_node.get("is_expanded")
    assert p3_node.get("is_expanded")
    
    # Collapse p2 (parent of p3)
    handler.collapse_node(p2_node["id"])
    
    # Verify p2 is collapsed
    full_graph = handler.get_full_graph_with_state()
    p2_node = next(n for n in full_graph["nodes"] if n.get("label") == "p2")
    assert not p2_node.get("is_expanded")
    
    # Verify p3 is ALSO collapsed (recursive behavior)
    p3_node = next(n for n in full_graph["nodes"] if n.get("label") == "p3")
    assert not p3_node.get("is_expanded"), "Child p3 should be collapsed when parent p2 is collapsed"

def test_ui_handler_full_graph_with_state():
    pipeline = create_nested_pipeline()
    handler = UIHandler(pipeline)
    handler.set_initial_depth(2)
    
    # Get full graph
    full_graph = handler.get_full_graph_with_state()
    nodes = full_graph["nodes"]
    
    # Should contain ALL nodes (even invisible ones at depth 2, like func_c)
    labels = [n.get("label") for n in nodes]
    assert "func_a" in labels
    assert "p2" in labels
    assert "func_b" in labels
    assert "p3" in labels
    assert "func_c" in labels
    
    # Check expansion flags
    # p2 should be expanded
    p2_node = next(n for n in nodes if n.get("label") == "p2")
    assert p2_node.get("is_expanded")
    
    # p3 should NOT be expanded
    p3_node = next(n for n in nodes if n.get("label") == "p3")
    assert not p3_node.get("is_expanded")

def create_complex_pipeline():
    """Create a pipeline with 2 nested pipelines, one having another child.
    
    Structure:
    Root (p1)
      -> Nested1 (p2)
           -> DeepNested (p3)
      -> Nested2 (p4)
    """
    # Level 3
    p3 = Pipeline(nodes=[func_c], name="DeepNested")
    
    # Level 2 (Nested1 contains DeepNested)
    n_p3 = p3.as_node(input_mapping={"b_out": "z"})
    p2 = Pipeline(nodes=[func_b, n_p3], name="Nested1")
    
    # Level 2 (Nested2 is simple)
    p4 = Pipeline(nodes=[func_b], name="Nested2")
    
    # Level 1 (Root contains Nested1 and Nested2)
    # Map outputs to avoid collision in Root
    n_p2 = p2.as_node(
        input_mapping={"a_out": "y"},
        output_mapping={"b_out": "b_out_1"}
    )
    n_p4 = p4.as_node(
        input_mapping={"a_out": "y"},
        output_mapping={"b_out": "b_out_2"}
    )
    
    p1 = Pipeline(nodes=[func_a, n_p2, n_p4], name="Root")
    return p1

def test_ui_handler_state_parity():
    """Verify parity between expanding from depth=1 and collapsing from depth=3."""
    pipeline = create_complex_pipeline()
    
    # Scenario 1: Start at depth=1 (all collapsed), then expand
    handler1 = UIHandler(pipeline)
    handler1.set_initial_depth(1)
    
    # Get IDs for Nested1 and DeepNested
    full_graph1 = handler1.get_full_graph_with_state()
    nested1_id = next(n["id"] for n in full_graph1["nodes"] if n.get("label") == "Nested1")
    deep_nested_id = next(n["id"] for n in full_graph1["nodes"] if n.get("label") == "DeepNested")
    
    # Expand Nested1 -> DeepNested
    handler1.expand_node(nested1_id)
    handler1.expand_node(deep_nested_id)
    
    # Scenario 2: Start at depth=3 (all expanded)
    handler2 = UIHandler(pipeline)
    handler2.set_initial_depth(3)
    
    # Verify initial state matches expected full expansion
    full_graph2 = handler2.get_full_graph_with_state()
    n1_node = next(n for n in full_graph2["nodes"] if n["id"] == nested1_id)
    dn_node = next(n for n in full_graph2["nodes"] if n["id"] == deep_nested_id)
    assert n1_node.get("is_expanded")
    assert dn_node.get("is_expanded")
    
    # Now collapse DeepNested, then collapse Nested1, then RE-EXPAND Nested1
    # Goal: Reach state where Nested1 is expanded but DeepNested is collapsed
    # This matches handler1 state if we hadn't expanded DeepNested yet.
    
    # Let's adjust the target state to be: Nested1 Expanded, DeepNested Collapsed, Nested2 Collapsed
    
    # Handler 1 (Depth 1 -> Expand Nested1)
    handler1.set_initial_depth(1)
    handler1.expand_node(nested1_id)
    # State 1: Root expanded (implicit), Nested1 expanded, DeepNested collapsed, Nested2 collapsed
    
    # Handler 2 (Depth 3 -> Collapse DeepNested -> Collapse Nested2)
    handler2.set_initial_depth(3)
    
    # Collapse DeepNested
    handler2.collapse_node(deep_nested_id)
    
    # Collapse Nested2
    nested2_id = next(n["id"] for n in full_graph2["nodes"] if n.get("label") == "Nested2")
    handler2.collapse_node(nested2_id)
    
    # State 2: Root expanded, Nested1 expanded (untouched), DeepNested collapsed, Nested2 collapsed
    
    # Compare expanded_nodes sets
    assert handler1.expanded_nodes == handler2.expanded_nodes
    
    # Further check: Collapse Nested1 in both
    handler1.collapse_node(nested1_id)
    handler2.collapse_node(nested1_id)
    
    assert handler1.expanded_nodes == handler2.expanded_nodes
    
    # Further check: Expand Nested1 in both. 
    # Since we implemented recursive collapse, DeepNested should be collapsed in BOTH.
    handler1.expand_node(nested1_id)
    handler2.expand_node(nested1_id)
    
    assert handler1.expanded_nodes == handler2.expanded_nodes
    
    # Verify DeepNested is NOT expanded in either
    assert deep_nested_id not in handler1.expanded_nodes
    assert deep_nested_id not in handler2.expanded_nodes


def test_ui_handler_event_diffs_are_minimal():
    pipeline = create_nested_pipeline()
    handler = UIHandler(pipeline, depth=1)

    # Expand the nested pipeline
    view_before = handler.get_view_data()
    p2_id = next(n["id"] for n in view_before["nodes"] if n.get("label") == "p2")

    expand_update = handler.handle_event({"type": "expand", "node_id": p2_id})

    added_labels = {n.get("label") for n in expand_update.added_nodes}
    assert {"func_b", "p3"} <= added_labels
    assert p2_id in {n["id"] for n in expand_update.updated_nodes}
    assert not expand_update.removed_nodes
    assert expand_update.view["nodes"]
    assert expand_update.view["edges"]

    # Collapse back and ensure removals only touch the nested nodes
    collapse_update = handler.handle_event({"type": "collapse", "node_id": p2_id})
    removed_ids = set(collapse_update.removed_nodes)
    assert removed_ids  # We should remove the nested nodes when collapsing
    assert p2_id in {n["id"] for n in collapse_update.updated_nodes}
    # No new additions when collapsing
    assert not collapse_update.added_nodes


def test_ui_handler_depth_none_expands_all():
    pipeline = create_nested_pipeline()
    handler = UIHandler(pipeline, depth=None)

    view = handler.get_view_data()
    labels = {n.get("label") for n in view["nodes"]}

    # All pipeline levels should be visible when depth is unlimited
    assert {"func_a", "p2", "func_b", "p3", "func_c"}.issubset(labels)


def test_simulate_event_does_not_mutate_state():
    pipeline = create_nested_pipeline()
    handler = UIHandler(pipeline, depth=1)
    initial_expanded = set(handler.expanded_nodes)
    view_before = handler.get_view_data()

    p2_id = next(n["id"] for n in view_before["nodes"] if n.get("label") == "p2")
    handler.simulate_event({"type": "expand", "node_id": p2_id})

    # State should be unchanged
    assert handler.expanded_nodes == initial_expanded
    assert handler.get_view_data() == view_before


def test_event_index_contains_pipeline_nodes():
    pipeline = create_nested_pipeline()
    handler = UIHandler(pipeline, depth=1)
    graph = handler.get_full_graph_with_state(include_events=True)

    event_index = graph.get("event_index") or {}
    p_nodes = [n for n in graph["nodes"] if n.get("node_type") == "PIPELINE"]
    for node in p_nodes:
        nid = node["id"]
        assert nid in event_index
        assert "expand" in event_index[nid]
        assert "collapse" in event_index[nid]
