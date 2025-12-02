"""Tests for branch node visualization."""

import pytest

from hypernodes import Pipeline, branch, node
from hypernodes.viz.graph_walker import GraphWalker
from hypernodes.viz.structures import BranchVizNode, VizEdge


class TestBranchVisualizationStructures:
    """Test that branch nodes create correct visualization structures."""

    def test_branch_creates_branch_viz_node(self):
        """Test that BranchNode creates BranchVizNode in graph walker."""

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        @node(output_name="result")
        def true_target(value: int) -> str:
            return "true"

        @node(output_name="result")
        def false_target(value: int) -> str:
            return "false"

        @branch(when_true=true_target, when_false=false_target)
        def is_positive(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, is_positive, true_target, false_target]
        )

        walker = GraphWalker(pipeline, expanded_nodes=set())
        graph = walker.get_visualization_data()

        # Find the branch node
        branch_nodes = [n for n in graph.nodes if isinstance(n, BranchVizNode)]
        assert len(branch_nodes) == 1

        branch_node = branch_nodes[0]
        assert branch_node.label == "is_positive"
        assert branch_node.function_name == "is_positive"
        assert branch_node.when_true_target == "true_target"
        assert branch_node.when_false_target == "false_target"

    def test_branch_creates_labeled_edges(self):
        """Test that branch node creates edges with True/False labels."""

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        @node(output_name="result")
        def true_target(value: int) -> str:
            return "true"

        @node(output_name="result")
        def false_target(value: int) -> str:
            return "false"

        @branch(when_true=true_target, when_false=false_target)
        def is_positive(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, is_positive, true_target, false_target]
        )

        walker = GraphWalker(pipeline, expanded_nodes=set())
        graph = walker.get_visualization_data()

        # Find edges from the branch node
        branch_edges = [e for e in graph.edges if e.source == "is_positive"]

        # Should have two edges: True and False
        assert len(branch_edges) == 2

        edge_labels = {e.label for e in branch_edges}
        assert edge_labels == {"True", "False"}

        # Check targets
        true_edge = next(e for e in branch_edges if e.label == "True")
        false_edge = next(e for e in branch_edges if e.label == "False")

        assert true_edge.target == "true_target"
        assert false_edge.target == "false_target"


class TestBranchJSVisualization:
    """Test JS visualization rendering for branch nodes."""

    def test_js_renderer_maps_branch_to_correct_type(self):
        """Test that JSRenderer maps BranchVizNode to BRANCH nodeType."""
        from hypernodes.viz.js.renderer import JSRenderer

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        @node(output_name="result")
        def true_target(value: int) -> str:
            return "true"

        @node(output_name="result")
        def false_target(value: int) -> str:
            return "false"

        @branch(when_true=true_target, when_false=false_target)
        def is_positive(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, is_positive, true_target, false_target]
        )

        walker = GraphWalker(pipeline, expanded_nodes=set())
        graph = walker.get_visualization_data()

        renderer = JSRenderer()
        rf_data = renderer.render(graph)

        # Find the branch node in React Flow data
        branch_rf_nodes = [
            n for n in rf_data["nodes"] if n.get("data", {}).get("nodeType") == "BRANCH"
        ]
        assert len(branch_rf_nodes) == 1

        branch_node = branch_rf_nodes[0]
        assert branch_node["data"]["label"] == "is_positive"
        assert branch_node["data"]["whenTrueTarget"] == "true_target"
        assert branch_node["data"]["whenFalseTarget"] == "false_target"

    def test_js_edges_have_labels(self):
        """Test that edges from branch have labels in JS data."""
        from hypernodes.viz.js.renderer import JSRenderer

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        @node(output_name="result")
        def true_target(value: int) -> str:
            return "true"

        @node(output_name="result")
        def false_target(value: int) -> str:
            return "false"

        @branch(when_true=true_target, when_false=false_target)
        def is_positive(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, is_positive, true_target, false_target]
        )

        walker = GraphWalker(pipeline, expanded_nodes=set())
        graph = walker.get_visualization_data()

        renderer = JSRenderer()
        rf_data = renderer.render(graph)

        # Find edges from branch
        branch_edges = [
            e for e in rf_data["edges"] if e["source"] == "is_positive"
        ]

        # Should have labels in data
        labels = {e.get("data", {}).get("label") for e in branch_edges}
        assert "True" in labels
        assert "False" in labels

    def test_pipeline_visualize_js_with_branch(self):
        """Test that pipeline.visualize() works with branch nodes."""

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        @node(output_name="result")
        def true_target(value: int) -> str:
            return "true"

        @node(output_name="result")
        def false_target(value: int) -> str:
            return "false"

        @branch(when_true=true_target, when_false=false_target)
        def is_positive(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, is_positive, true_target, false_target]
        )

        # Should not raise
        html = pipeline.visualize(engine="js")
        assert len(html) > 0
        assert "is_positive" in html
        assert "BRANCH" in html


class TestBranchGraphvizVisualization:
    """Test Graphviz visualization rendering for branch nodes."""

    def test_graphviz_renders_branch_as_diamond(self):
        """Test that Graphviz renders branch node with diamond shape."""
        from hypernodes.viz.graphviz.renderer import GraphvizRenderer

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        @node(output_name="result")
        def true_target(value: int) -> str:
            return "true"

        @node(output_name="result")
        def false_target(value: int) -> str:
            return "false"

        @branch(when_true=true_target, when_false=false_target)
        def is_positive(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, is_positive, true_target, false_target]
        )

        walker = GraphWalker(pipeline, expanded_nodes=set())
        graph = walker.get_visualization_data()

        renderer = GraphvizRenderer()
        # Check the DOT source before rendering to SVG
        renderer.graph_data = graph
        renderer.lines = ["digraph G {"]
        renderer._indent_level = 1
        renderer.nodes_by_parent = {}
        renderer.outputs_by_source = {}

        for n in graph.nodes:
            renderer.nodes_by_parent.setdefault(n.parent_id, []).append(n)

        renderer._render_scope(None)

        dot_source = "\n".join(renderer.lines)

        # Check that branch node uses diamond shape
        assert 'shape="diamond"' in dot_source
        assert "is_positive" in dot_source

    def test_graphviz_edge_labels(self):
        """Test that Graphviz renders edge labels for branch edges."""
        from hypernodes.viz.graphviz.renderer import GraphvizRenderer

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        @node(output_name="result")
        def true_target(value: int) -> str:
            return "true"

        @node(output_name="result")
        def false_target(value: int) -> str:
            return "false"

        @branch(when_true=true_target, when_false=false_target)
        def is_positive(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, is_positive, true_target, false_target]
        )

        walker = GraphWalker(pipeline, expanded_nodes=set())
        graph = walker.get_visualization_data()

        renderer = GraphvizRenderer()
        renderer.graph_data = graph
        renderer.lines = ["digraph G {"]
        renderer._indent_level = 1
        renderer.nodes_by_parent = {}
        renderer.outputs_by_source = {}

        for n in graph.nodes:
            renderer.nodes_by_parent.setdefault(n.parent_id, []).append(n)

        renderer._render_scope(None)

        # Render edges
        node_map = {n.id: n for n in graph.nodes}
        for edge in graph.edges:
            renderer._render_edge(edge, node_map)

        dot_source = "\n".join(renderer.lines)

        # Check for labeled edges
        assert 'label="True"' in dot_source
        assert 'label="False"' in dot_source

