"""Test script to verify uniform node width and type hint truncation.

This script generates an HTML visualization with various type hint lengths
to verify that:
1. All nodes have consistent max width (280px)
2. Type hints > 25 chars are truncated with "..."
3. The TEXTS debug tab shows truncation status

Usage:
    uv run python scripts/test_truncation_uniform.py
    
Then open outputs/test_truncation_uniform.html in a browser and:
1. Click the bug icon to enable debug mode
2. Click the TEXTS tab to see text analysis
3. Verify all nodes have consistent widths
4. Console: HyperNodesVizState.debug.getTextAnalysis()
"""

from typing import Dict, List, Tuple

from hypernodes import Pipeline, node
from hypernodes.viz import UIHandler
from hypernodes.viz.js.html_generator import generate_widget_html
from hypernodes.viz.js.renderer import JSRenderer


# Custom types for testing
class VectorStore:
    pass


class EmbeddingModelConfiguration:
    """A very long class name to test truncation."""
    pass


class SuperLongTypeNameThatDefinitelyExceedsTwentyFiveCharacters:
    """Extremely long type name."""
    pass


# Test nodes with various type hint lengths
@node(output_name="result")
def short_types(x: int, y: str) -> int:
    """Short type hints (< 10 chars)."""
    return x


@node(output_name="store")
def medium_types(query: str, vector_store: VectorStore) -> str:
    """Medium type hints (~11 chars)."""
    return ""


@node(output_name="config")
def long_type_input(model_config: EmbeddingModelConfiguration) -> str:
    """Long type hint (~28 chars) - should be truncated."""
    return ""


@node(output_name="data")
def very_long_type_input(
    processor: SuperLongTypeNameThatDefinitelyExceedsTwentyFiveCharacters,
) -> str:
    """Very long type hint (~50 chars) - definitely truncated."""
    return ""


@node(output_name="embeddings")
def dict_type_output(text: str) -> Dict[str, List[float]]:
    """Output with Dict[str, List[float]] (~22 chars)."""
    return {}


@node(output_name=("scores", "metadata"))
def multi_output_long_types(data: list) -> Tuple[List[float], Dict[str, int]]:
    """Multiple outputs with long types."""
    return [], {}


@node(output_name="processed")
def combined_long(
    items: List[Dict[str, int]], config: EmbeddingModelConfiguration
) -> Dict[str, List[str]]:
    """Both input and output have long types."""
    return {}


def main():
    # Create pipeline with various type hint lengths
    pipeline = Pipeline(
        nodes=[
            short_types,
            medium_types,
            long_type_input,
            very_long_type_input,
            dict_type_output,
            multi_output_long_types,
            combined_long,
        ],
        name="truncation_test",
    )

    print("=== Uniform Width & Truncation Test ===\n")

    # Test with show_types=True, separate_outputs=True
    print("Generating: show_types=True, separate_outputs=True")
    handler = UIHandler(pipeline, depth=99)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)
    renderer = JSRenderer()
    rf_data = renderer.render(
        graph_data, theme="dark", separate_outputs=True, show_types=True
    )

    # Analyze node data
    print("\n--- Node Analysis ---")
    for n in rf_data["nodes"]:
        node_id = n.get("id", "")
        node_type = n.get("data", {}).get("nodeType", "")
        label = n.get("data", {}).get("label", "")
        type_hint = n.get("data", {}).get("typeHint", "")

        if type_hint:
            truncated = len(type_hint) > 25
            status = "[TRUNCATED]" if truncated else "[OK]"
            print(f"{status} {node_type}: {label}")
            print(f"         typeHint: '{type_hint}' ({len(type_hint)} chars)")

    html = generate_widget_html(rf_data)
    with open("outputs/test_truncation_uniform.html", "w") as f:
        f.write(html)
    print("\nSaved: outputs/test_truncation_uniform.html")

    # Also generate combined outputs mode
    print("\n\nGenerating: show_types=True, separate_outputs=False")
    rf_data_combined = renderer.render(
        graph_data, theme="dark", separate_outputs=False, show_types=True
    )
    html_combined = generate_widget_html(rf_data_combined)
    with open("outputs/test_truncation_combined.html", "w") as f:
        f.write(html_combined)
    print("Saved: outputs/test_truncation_combined.html")

    # Print summary
    print("\n=== Summary ===")
    print("- TYPE_HINT_MAX_CHARS = 25")
    print("- MAX_NODE_WIDTH = 280px")
    print("\nTo verify:")
    print("1. Open the HTML files in a browser")
    print("2. Click the bug icon (debug mode)")
    print("3. Click the TEXTS tab")
    print("4. Check that 'Longest' column shows truncated types in red")
    print("5. Run in console: HyperNodesVizState.debug.getTextAnalysis()")


if __name__ == "__main__":
    main()

