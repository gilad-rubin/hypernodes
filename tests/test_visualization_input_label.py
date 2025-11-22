"""Test that input node labels don't show 'input_' prefix."""

import pytest

from hypernodes import Pipeline, node


class TestInputNodeLabels:
    """Test that input nodes show clean labels without 'input_' prefix."""

    def test_simple_input_label_no_prefix(self):
        """Input node label should be 'llm', not 'input_llm'."""
        @node(output_name="result")
        def process(llm: str) -> str:
            return llm.upper()

        pipeline = Pipeline(nodes=[process], name="test")

        # Get the render data
        from hypernodes.viz.ui_handler import UIHandler
        from hypernodes.viz.structures import DataNode

        handler = UIHandler(pipeline, depth=1, group_inputs=False)
        viz_graph = handler.get_visualization_data()

        # Find the input node (DataNode with no source_id)
        input_nodes = [n for n in viz_graph.nodes if isinstance(n, DataNode) and n.source_id is None]
        assert len(input_nodes) == 1, "Should have exactly one input node"

        input_node = input_nodes[0]
        assert "llm" in input_node.id, f"Node ID should contain 'llm', got {input_node.id}"
        # Name should be the parameter name, not prefixed with "input_"
        assert input_node.name == "llm", f"Name should be 'llm', got '{input_node.name}'"
        assert not input_node.name.startswith("input_"), f"Name should not start with 'input_', got '{input_node.name}'"

    def test_nested_pipeline_input_label_no_prefix(self):
        """Test input labels in nested pipelines don't show prefix."""
        @node(output_name="inner_result")
        def inner_process(model: str) -> str:
            return model.lower()

        @node(output_name="final")
        def outer_process(inner_result: str) -> str:
            return inner_result.upper()

        inner = Pipeline(nodes=[inner_process], name="InnerPipeline")
        inner_node = inner.as_node(name="InnerPipeline")
        outer = Pipeline(nodes=[inner_node, outer_process], name="OuterPipeline")

        # Check depth=2 (expanded)
        from hypernodes.viz.ui_handler import UIHandler
        from hypernodes.viz.structures import DataNode

        handler = UIHandler(outer, depth=2, group_inputs=False)
        viz_graph = handler.get_visualization_data()

        # Find input nodes (DataNode with no source_id)
        input_nodes = [n for n in viz_graph.nodes if isinstance(n, DataNode) and n.source_id is None]
        assert len(input_nodes) > 0, "Should have at least one input node"

        for input_node in input_nodes:
            # Name should be the parameter name without prefix
            param_name = input_node.name
            assert not param_name.startswith("input_"), \
                f"Name should not start with 'input_', got '{param_name}'"

    def test_bound_input_label_no_prefix(self):
        """Bound input labels should also not show prefix."""
        pytest.importorskip("graphviz")

        @node(output_name="result")
        def process(data: str, multiplier: int = 1) -> str:
            return data * multiplier

        pipeline = Pipeline(nodes=[process], name="test").bind(multiplier=5)

        from hypernodes.viz.ui_handler import UIHandler
        from hypernodes.viz.structures import DataNode

        handler = UIHandler(pipeline, depth=1, group_inputs=False)
        viz_graph = handler.get_visualization_data()

        # Find input nodes (DataNode with no source_id)
        input_nodes = [n for n in viz_graph.nodes if isinstance(n, DataNode) and n.source_id is None]

        for input_node in input_nodes:
            # Name should be the parameter name without prefix
            assert not input_node.name.startswith("input_"), \
                f"Name should not start with 'input_', got '{input_node.name}'"

    def test_input_group_label_no_prefix(self):
        """Grouped input labels should not show prefix for individual params."""
        pytest.importorskip("graphviz")

        @node(output_name="result")
        def process(a: int, b: int, c: int) -> int:
            return a + b + c

        pipeline = Pipeline(nodes=[process], name="test")

        from hypernodes.viz.ui_handler import UIHandler
        from hypernodes.viz.structures import GroupDataNode

        handler = UIHandler(pipeline, depth=1, group_inputs=True)
        viz_graph = handler.get_visualization_data()

        # Find grouped input nodes
        group_nodes = [n for n in viz_graph.nodes if isinstance(n, GroupDataNode)]

        if group_nodes:
            # If we have a group, check the individual node names
            group_node = group_nodes[0]
            param_names = [dn.name for dn in group_node.nodes]
            
            # Names should be clean param names without "input_" prefix
            for name in param_names:
                assert not name.startswith("input_"), f"Name should not start with 'input_', got '{name}'"
            
            # Should have the actual param names
            assert "a" in param_names, f"Should have 'a' in {param_names}"
            assert "b" in param_names, f"Should have 'b' in {param_names}"
            assert "c" in param_names, f"Should have 'c' in {param_names}"
