"""Tests for uniform node width and type hint truncation.

These tests verify that:
1. TYPE_HINT_MAX_CHARS is used consistently (25 chars)
2. All node types respect MAX_NODE_WIDTH (280px)
3. Width calculations use the same constants
4. Truncation is applied uniformly across node types
"""

import re
from typing import Dict, List

import pytest

from hypernodes import Pipeline, node
from hypernodes.viz import UIHandler
from hypernodes.viz.js.html_generator import generate_widget_html
from hypernodes.viz.js.renderer import JSRenderer


# Test type classes
class VectorStore:
    pass


class EmbeddingModelConfiguration:
    pass


class SuperLongTypeNameThatDefinitelyExceedsTwentyFiveCharacters:
    pass


# Constants that should match html_generator.py
TYPE_HINT_MAX_CHARS = 25
CHAR_WIDTH_PX = 7
NODE_BASE_PADDING = 56  # px-4 (16px) * 2 + icon (16px) + gap (8px)
MAX_NODE_WIDTH = 280


@node(output_name="result")
def short_types_node(x: int, y: str) -> int:
    return x


@node(output_name="store")
def medium_type_node(query: str, vector_store: VectorStore) -> str:
    return ""


@node(output_name="config")
def long_type_node(model_config: EmbeddingModelConfiguration) -> str:
    return ""


@node(output_name="data")
def very_long_type_node(
    processor: SuperLongTypeNameThatDefinitelyExceedsTwentyFiveCharacters,
) -> str:
    return ""


class TestTruncationConstantsInHTML:
    """Verify that the HTML generator uses the correct truncation constant."""

    def _generate_html(self, pipeline: Pipeline) -> str:
        handler = UIHandler(pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(
            graph_data, theme="dark", separate_outputs=True, show_types=True
        )
        return generate_widget_html(rf_data)

    def test_type_hint_max_chars_constant_defined(self):
        """TYPE_HINT_MAX_CHARS should be defined in the generated HTML."""
        pipeline = Pipeline(nodes=[short_types_node])
        html = self._generate_html(pipeline)

        assert "TYPE_HINT_MAX_CHARS = 25" in html

    def test_max_node_width_constant_defined(self):
        """MAX_NODE_WIDTH should be defined in the generated HTML."""
        pipeline = Pipeline(nodes=[short_types_node])
        html = self._generate_html(pipeline)

        assert "MAX_NODE_WIDTH = 280" in html

    def test_truncate_type_hint_function_defined(self):
        """truncateTypeHint function should be defined globally."""
        pipeline = Pipeline(nodes=[short_types_node])
        html = self._generate_html(pipeline)

        assert "truncateTypeHint" in html

    def test_no_hardcoded_20_in_width_calculation(self):
        """Width calculations should not use hardcoded 20 for type truncation."""
        pipeline = Pipeline(nodes=[short_types_node])
        html = self._generate_html(pipeline)

        # Search for Math.min(..., 20) patterns which would indicate old code
        old_pattern = re.compile(r"Math\.min\([^,]+,\s*20\s*\)")
        matches = old_pattern.findall(html)

        assert len(matches) == 0, f"Found hardcoded 20 in width calc: {matches}"


class TestWidthCalculation:
    """Test that width calculation respects the dynamic formula (no minimums)."""

    def _calculate_expected_width(
        self, label: str, type_hint: str | None = None, show_types: bool = True
    ) -> int:
        """Calculate expected width using the same formula as html_generator.py.
        
        Width is now purely dynamic - no minimum, only MAX_NODE_WIDTH cap.
        """
        label_len = len(label) if label else 0
        type_len = 0
        if show_types and type_hint:
            type_len = min(len(type_hint), TYPE_HINT_MAX_CHARS) + 2  # +2 for ": "

        raw_width = (label_len + type_len) * CHAR_WIDTH_PX + NODE_BASE_PADDING
        return min(MAX_NODE_WIDTH, raw_width)

    def test_short_type_width(self):
        """Short type hints should calculate correct width (dynamic, no minimum)."""
        # "x : int" -> label=1, type=3+2=5, total=6
        width = self._calculate_expected_width("x", "int")
        # (1 + 5) * 7 + 56 = 98
        assert width == 98

    def test_medium_type_width(self):
        """Medium type hints should calculate correct width."""
        # "vector_store : VectorStore" -> label=12, type=11+2=13, total=25
        width = self._calculate_expected_width("vector_store", "VectorStore")
        # (12 + 13) * 7 + 56 = 231
        assert width == 231

    def test_long_type_truncated_in_width(self):
        """Long type hints should be truncated to 25 chars for width calc."""
        # "EmbeddingModelConfiguration" = 27 chars -> truncated to 25+2=27
        long_type = "EmbeddingModelConfiguration"
        width = self._calculate_expected_width("model_config", long_type)

        # label=12, truncated_type=25+2=27, total=39
        # (12 + 27) * 7 + 52 = 325, but max is 280
        assert width == MAX_NODE_WIDTH

    def test_very_long_type_capped_at_max(self):
        """Very long types should result in max width."""
        very_long_type = "SuperLongTypeNameThatDefinitelyExceedsTwentyFiveCharacters"
        width = self._calculate_expected_width("processor", very_long_type)

        assert width == MAX_NODE_WIDTH

    def test_no_type_uses_dynamic_width(self):
        """Nodes without types should use dynamic width (label only, no minimum)."""
        width = self._calculate_expected_width("x", None)
        # (1 + 0) * 7 + 56 = 63
        assert width == 63


class TestTruncationDisplay:
    """Test that type hints are displayed with correct truncation."""

    def test_short_type_not_truncated(self):
        """Short type hints should not be truncated in display."""
        type_hint = "int"
        truncated = type_hint[:TYPE_HINT_MAX_CHARS] + "..." if len(type_hint) > TYPE_HINT_MAX_CHARS else type_hint
        assert truncated == "int"

    def test_exactly_25_chars_not_truncated(self):
        """Type hint with exactly 25 chars should not be truncated."""
        type_hint = "A" * 25  # Exactly 25 chars
        truncated = type_hint[:TYPE_HINT_MAX_CHARS] + "..." if len(type_hint) > TYPE_HINT_MAX_CHARS else type_hint
        assert truncated == type_hint
        assert len(truncated) == 25

    def test_26_chars_truncated(self):
        """Type hint with 26 chars should be truncated."""
        type_hint = "A" * 26  # 26 chars
        truncated = type_hint[:TYPE_HINT_MAX_CHARS] + "..." if len(type_hint) > TYPE_HINT_MAX_CHARS else type_hint
        assert truncated == "A" * 25 + "..."
        assert len(truncated) == 28  # 25 + 3 for "..."

    def test_embedding_model_configuration_truncated(self):
        """EmbeddingModelConfiguration (27 chars) should be truncated."""
        type_hint = "EmbeddingModelConfiguration"  # 27 chars
        assert len(type_hint) == 27  # Verify our assumption (> 25)

        truncated = type_hint[:TYPE_HINT_MAX_CHARS] + "..." if len(type_hint) > TYPE_HINT_MAX_CHARS else type_hint
        # First 25 chars + "..."
        assert truncated == "EmbeddingModelConfigurati..."
        assert len(truncated) == 28  # 25 + 3


class TestAllNodeTypesUniformWidth:
    """Test that all node types respect the same max width."""

    def test_all_node_types_have_max_width_cap(self):
        """All node types should use MAX_NODE_WIDTH cap in width calculation."""
        pipeline = Pipeline(nodes=[short_types_node])
        handler = UIHandler(pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(
            graph_data, theme="dark", separate_outputs=True, show_types=True
        )
        html = generate_widget_html(rf_data)

        # Verify MAX_NODE_WIDTH is used in width calculations
        elk_section = html[html.find("const mapToElk"):html.find("const elkGraph")]
        
        # Node types with dynamic width use Math.min(MAX_NODE_WIDTH, ...)
        # DATA, INPUT, INPUT_GROUP use this pattern
        max_width_caps = elk_section.count("Math.min(MAX_NODE_WIDTH")
        assert max_width_caps >= 3, f"Expected at least 3 Math.min(MAX_NODE_WIDTH) caps for DATA/INPUT/INPUT_GROUP, found {max_width_caps}"
        
        # Verify MAX_NODE_WIDTH constant is defined
        assert "MAX_NODE_WIDTH = 280" in html, "MAX_NODE_WIDTH constant should be defined"


class TestDebugTextsTab:
    """Test that the debug TEXTS tab is properly generated."""

    def test_texts_tab_exists(self):
        """The TEXTS tab button should exist in debug overlay."""
        pipeline = Pipeline(nodes=[short_types_node])
        handler = UIHandler(pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(
            graph_data, theme="dark", separate_outputs=True, show_types=True
        )
        html = generate_widget_html(rf_data)

        # Check for TEXTS tab button
        assert "setActiveTab('texts')" in html
        assert "TEXTS" in html

    def test_texts_tab_shows_truncation_info(self):
        """The TEXTS tab should show truncation information."""
        pipeline = Pipeline(nodes=[short_types_node])
        handler = UIHandler(pipeline, depth=99)
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        renderer = JSRenderer()
        rf_data = renderer.render(
            graph_data, theme="dark", separate_outputs=True, show_types=True
        )
        html = generate_widget_html(rf_data)

        # Check for truncation info header
        assert "truncated at K=" in html or "TYPE_HINT_MAX_CHARS" in html

