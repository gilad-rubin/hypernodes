
import pytest

from hypernodes import Pipeline
from hypernodes.node import node


def test_graphviz_output_cleanliness():
    """Test that Graphviz output does not contain internal implementation artifacts."""
    
    @node(output_name="summed")
    def add(x: int, y: int) -> int:
        return x + y
        
    @node(output_name="retrieved")
    def retrieve(x: int) -> int:
        return x
    
    pipeline = Pipeline(nodes=[add, retrieve])
    
    # Generate visualization HTML
    result = pipeline.visualize()
    
    # Extract HTML content
    if hasattr(result, 'data'):
        html_str = str(result.data)
    elif hasattr(result, '_repr_html_'):
        html_str = str(result._repr_html_())
    else:
        html_str = str(result)
        
    # Check for unwanted artifacts - synthetic nodes that don't exist in the pipeline
    unwanted_patterns = [
        "<title>group_inputs</title>",
        "<title>fn_retrieve</title>",
        "<title>Grouped Inputs</title>",  # GroupDataNode should not exist
        "GroupDataNode",
    ]
    
    found_artifacts = []
    for pattern in unwanted_patterns:
        if pattern in html_str:
            # Find context
            idx = html_str.find(pattern)
            start = max(0, idx - 50)
            end = min(len(html_str), idx + 50)
            context = html_str[start:end]
            found_artifacts.append(f"{pattern} in context: ...{context}...")
            
    if found_artifacts:
        pytest.fail(
            "Found unwanted internal artifacts in Graphviz output:\n" + "\n".join(found_artifacts) +
            "\nThe visualization should only show user-defined nodes (x, y, add, retrieve), not synthetic grouped nodes."
        )
        
    # Verify that we have the actual pipeline nodes
    expected_titles = [
        "<title>retrieve</title>",
        "<title>add</title>",
        "<title>x</title>",
        "<title>y</title>",
    ]
    
    missing_titles = []
    for title in expected_titles:
        if title not in html_str:
            missing_titles.append(title)
            
    if missing_titles:
        pytest.fail(
            f"Missing expected titles in Graphviz output: {missing_titles}\n"
            "The visualization should show all actual pipeline nodes."
        )
