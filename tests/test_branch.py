"""Tests for branch nodes and conditional execution."""

import pytest

from hypernodes import Pipeline, branch, node
from hypernodes.branch import BranchNode
from hypernodes.callbacks import CallbackContext, PipelineCallback
from hypernodes.exceptions import DependencyError


class TestBranchNodeBasics:
    """Test basic BranchNode creation and properties."""

    def test_branch_decorator_creates_branch_node(self):
        """Test that @branch decorator creates a BranchNode."""

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        @node(output_name="positive_result")
        def process_positive(value: int) -> str:
            return f"Positive: {value}"

        @node(output_name="negative_result")
        def process_negative(value: int) -> str:
            return f"Non-positive: {value}"

        @branch(when_true=process_positive, when_false=process_negative)
        def is_positive(value: int) -> bool:
            return value > 0

        assert isinstance(is_positive, BranchNode)
        assert is_positive.name == "is_positive"
        assert is_positive.when_true_name == "process_positive"
        assert is_positive.when_false_name == "process_negative"

    def test_branch_node_output_names(self):
        """Test that branch node produces gate signal output names."""

        @node(output_name="result")
        def target_a(x: int) -> int:
            return x

        @node(output_name="result")
        def target_b(x: int) -> int:
            return x

        @branch(when_true=target_a, when_false=target_b)
        def my_branch(x: int) -> bool:
            return x > 0

        assert my_branch.output_name == ("_gate_my_branch_true", "_gate_my_branch_false")
        assert my_branch.true_gate == "_gate_my_branch_true"
        assert my_branch.false_gate == "_gate_my_branch_false"

    def test_branch_node_root_args(self):
        """Test that branch node correctly extracts root_args from function."""

        @node(output_name="result")
        def dummy(x: int) -> int:
            return x

        @branch(when_true=dummy, when_false=dummy)
        def check_multiple(a: int, b: str, c: float) -> bool:
            return a > 0

        assert check_multiple.root_args == ("a", "b", "c")

    def test_branch_node_callable(self):
        """Test that branch node can be called directly."""

        @node(output_name="result")
        def dummy(x: int) -> int:
            return x

        @branch(when_true=dummy, when_false=dummy)
        def is_even(value: int) -> bool:
            return value % 2 == 0

        assert is_even(value=4) is True
        assert is_even(value=3) is False


class TestBranchExecution:
    """Test branch node execution with SeqEngine."""

    def test_basic_branch_true_path(self):
        """Test that True path executes when condition is True."""

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        @node(output_name="result")
        def process_positive(value: int) -> str:
            return f"Positive: {value}"

        @node(output_name="result")
        def process_negative(value: int) -> str:
            return f"Non-positive: {value}"

        @branch(when_true=process_positive, when_false=process_negative)
        def is_positive(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, is_positive, process_positive, process_negative]
        )
        result = pipeline.run(inputs={"x": 5})

        assert result["value"] == 5
        assert result["result"] == "Positive: 5"

    def test_basic_branch_false_path(self):
        """Test that False path executes when condition is False."""

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        @node(output_name="result")
        def process_positive(value: int) -> str:
            return f"Positive: {value}"

        @node(output_name="result")
        def process_negative(value: int) -> str:
            return f"Non-positive: {value}"

        @branch(when_true=process_positive, when_false=process_negative)
        def is_positive(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, is_positive, process_positive, process_negative]
        )
        result = pipeline.run(inputs={"x": -3})

        assert result["value"] == -3
        assert result["result"] == "Non-positive: -3"

    def test_branch_with_multiple_inputs(self):
        """Test branch node that takes multiple inputs."""

        @node(output_name="confidence")
        def get_confidence(data: dict) -> float:
            return data.get("confidence", 0.0)

        @node(output_name="threshold")
        def get_threshold(config: dict) -> float:
            return config.get("threshold", 0.5)

        @node(output_name="result")
        def process_high(confidence: float) -> str:
            return f"High confidence: {confidence}"

        @node(output_name="result")
        def process_low(confidence: float) -> str:
            return f"Low confidence: {confidence}"

        @branch(when_true=process_high, when_false=process_low)
        def should_proceed(confidence: float, threshold: float) -> bool:
            return confidence >= threshold

        pipeline = Pipeline(
            nodes=[
                get_confidence,
                get_threshold,
                should_proceed,
                process_high,
                process_low,
            ]
        )

        # High confidence
        result = pipeline.run(
            inputs={"data": {"confidence": 0.8}, "config": {"threshold": 0.5}}
        )
        assert result["result"] == "High confidence: 0.8"

        # Low confidence
        result = pipeline.run(
            inputs={"data": {"confidence": 0.3}, "config": {"threshold": 0.5}}
        )
        assert result["result"] == "Low confidence: 0.3"

    def test_branch_with_downstream_chain(self):
        """Test branch where each path has multiple downstream nodes.
        
        Note: Each branch path produces unique output names - merging outputs
        from different branch paths requires the same output name on the
        direct branch targets (not downstream nodes).
        """

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        @node(output_name="doubled")
        def double_positive(value: int) -> int:
            return value * 2

        @node(output_name="positive_result")
        def format_positive(doubled: int) -> str:
            return f"Result: {doubled}"

        @node(output_name="abs_value")
        def negate_negative(value: int) -> int:
            return abs(value)

        @node(output_name="negative_result")
        def format_negative(abs_value: int) -> str:
            return f"Absolute: {abs_value}"

        @branch(when_true=double_positive, when_false=negate_negative)
        def is_positive(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[
                get_value,
                is_positive,
                double_positive,
                format_positive,
                negate_negative,
                format_negative,
            ]
        )

        # Positive path: 5 -> 10 -> "Result: 10"
        result = pipeline.run(inputs={"x": 5})
        assert result["positive_result"] == "Result: 10"
        assert "negative_result" not in result  # Not executed

        # Negative path: -7 -> 7 -> "Absolute: 7"
        result = pipeline.run(inputs={"x": -7})
        assert result["negative_result"] == "Absolute: 7"
        assert "positive_result" not in result  # Not executed


class TestBranchValidation:
    """Test branch node validation during pipeline construction."""

    def test_missing_true_target_raises_error(self):
        """Test that missing True target raises DependencyError."""

        @node(output_name="result")
        def existing_node(x: int) -> int:
            return x

        @branch(when_true=existing_node, when_false=existing_node)
        def my_branch(x: int) -> bool:
            return x > 0

        # Remove the existing_node from pipeline - only include a different node
        @node(output_name="other")
        def other_node(x: int) -> int:
            return x

        with pytest.raises(DependencyError, match="targets.*when_true"):
            Pipeline(nodes=[my_branch, other_node])

    def test_missing_false_target_raises_error(self):
        """Test that missing False target raises DependencyError."""

        @node(output_name="result")
        def true_target(x: int) -> int:
            return x

        @node(output_name="other")
        def false_target(x: int) -> int:
            return x

        @branch(when_true=true_target, when_false=false_target)
        def my_branch(x: int) -> bool:
            return x > 0

        # Only include true_target, not false_target
        with pytest.raises(DependencyError, match="targets.*when_false"):
            Pipeline(nodes=[my_branch, true_target])


class TestBranchCallbacks:
    """Test branch-related callback hooks."""

    def test_on_branch_decision_callback(self):
        """Test that on_branch_decision is called when branch executes."""
        decisions = []

        class TrackingCallback(PipelineCallback):
            def on_branch_decision(
                self, branch_id: str, decision: bool, ctx: CallbackContext
            ):
                decisions.append((branch_id, decision))

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
        def my_branch(value: int) -> bool:
            return value > 0

        from hypernodes import SeqEngine

        engine = SeqEngine(callbacks=[TrackingCallback()])
        pipeline = Pipeline(
            nodes=[get_value, my_branch, true_target, false_target], engine=engine
        )

        pipeline.run(inputs={"x": 5})
        assert ("my_branch", True) in decisions

        decisions.clear()
        pipeline.run(inputs={"x": -5})
        assert ("my_branch", False) in decisions

    def test_on_node_skipped_callback(self):
        """Test that on_node_skipped is called for skipped nodes."""
        skipped = []

        class TrackingCallback(PipelineCallback):
            def on_node_skipped(
                self, node_id: str, reason: str, ctx: CallbackContext
            ):
                skipped.append(node_id)

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
        def my_branch(value: int) -> bool:
            return value > 0

        from hypernodes import SeqEngine

        engine = SeqEngine(callbacks=[TrackingCallback()])
        pipeline = Pipeline(
            nodes=[get_value, my_branch, true_target, false_target], engine=engine
        )

        # True path -> false_target should be skipped
        pipeline.run(inputs={"x": 5})
        assert "false_target" in skipped

        skipped.clear()
        # False path -> true_target should be skipped
        pipeline.run(inputs={"x": -5})
        assert "true_target" in skipped


class TestBranchGraphBuilder:
    """Test graph builder handling of branch nodes."""

    def test_branch_gates_tracked_in_graph(self):
        """Test that branch gates are tracked in GraphResult."""

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
        def my_branch(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, my_branch, true_target, false_target]
        )

        assert "_gate_my_branch_true" in pipeline.graph.branch_gates
        assert "_gate_my_branch_false" in pipeline.graph.branch_gates

    def test_gate_dependencies_tracked(self):
        """Test that target nodes have gate dependencies."""

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
        def my_branch(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, my_branch, true_target, false_target]
        )

        # true_target should depend on true gate
        true_target_node = [n for n in pipeline.nodes if n.name == "true_target"][0]
        assert "_gate_my_branch_true" in pipeline.graph.gate_dependencies.get(
            true_target_node, []
        )

        # false_target should depend on false gate
        false_target_node = [n for n in pipeline.nodes if n.name == "false_target"][0]
        assert "_gate_my_branch_false" in pipeline.graph.gate_dependencies.get(
            false_target_node, []
        )

    def test_exclusive_producers_tracked(self):
        """Test that nodes producing same output in exclusive branches are tracked."""

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
        def my_branch(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, my_branch, true_target, false_target]
        )

        # Both nodes produce "result" - should be tracked as exclusive
        assert "result" in pipeline.graph.exclusive_producers
        producers = pipeline.graph.exclusive_producers["result"]
        producer_names = [p.name for p in producers]
        assert "true_target" in producer_names
        assert "false_target" in producer_names


class TestBranchWithNestedPipelines:
    """Test branch nodes with nested pipelines."""

    def test_branch_inside_nested_pipeline(self):
        """Test that branch works inside a nested pipeline."""

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        @node(output_name="result")
        def positive_handler(value: int) -> str:
            return f"Positive: {value}"

        @node(output_name="result")
        def negative_handler(value: int) -> str:
            return f"Negative: {value}"

        @branch(when_true=positive_handler, when_false=negative_handler)
        def is_positive(value: int) -> bool:
            return value > 0

        # Inner pipeline with branch
        inner = Pipeline(
            nodes=[get_value, is_positive, positive_handler, negative_handler]
        )

        @node(output_name="final")
        def wrap_result(result: str) -> str:
            return f"[{result}]"

        # Outer pipeline
        outer = Pipeline(nodes=[inner.as_node(), wrap_result])

        result = outer.run(inputs={"x": 5})
        assert result["final"] == "[Positive: 5]"

        result = outer.run(inputs={"x": -3})
        assert result["final"] == "[Negative: -3]"

    def test_branch_targeting_nested_pipeline(self):
        """Test branch that targets a nested pipeline node."""

        @node(output_name="value")
        def get_value(x: int) -> int:
            return x

        # Simple inner pipeline
        @node(output_name="doubled")
        def double_it(value: int) -> int:
            return value * 2

        inner = Pipeline(nodes=[double_it])

        @node(output_name="result")
        def negate_it(value: int) -> int:
            return -value

        @branch(when_true=inner.as_node(), when_false=negate_it)
        def is_positive(value: int) -> bool:
            return value > 0

        pipeline = Pipeline(
            nodes=[get_value, is_positive, inner.as_node(), negate_it]
        )

        # Positive path goes through inner pipeline
        result = pipeline.run(inputs={"x": 5})
        assert result["doubled"] == 10

        # Negative path goes through negate_it
        result = pipeline.run(inputs={"x": -5})
        assert result["result"] == 5  # -(-5) = 5

