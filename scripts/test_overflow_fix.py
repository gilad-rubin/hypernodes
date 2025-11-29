"""Test script to verify the overflow fix with the VectorStore example."""

from typing import Any, Dict, List, Optional, Tuple

from hypernodes import Pipeline, node
from hypernodes.viz import UIHandler
from hypernodes.viz.js.html_generator import generate_widget_html
from hypernodes.viz.js.renderer import JSRenderer


class VectorStore:
    """Placeholder class for type hints."""
    pass


@node(output_name="store")
def retrieve_with_store(vector_store: VectorStore, top_k: int) -> str:
    """Example matching the screenshot issue - VectorStore type."""
    return "result"


@node(output_name="embeddings")
def get_embeddings(text: str) -> Dict[str, List[float]]:
    """Node with a very long type hint."""
    return {}


@node(output_name="results")
def complex_return(items: List[Dict[str, Any]]) -> Tuple[List[int], Dict[str, float], Optional[str]]:
    """Node with complex nested type hints."""
    return ([], {}, None)


def main():
    # Create pipeline with nodes that have long type hints
    pipeline = Pipeline(nodes=[retrieve_with_store, get_embeddings, complex_return])
    
    # Generate JS visualization
    handler = UIHandler(pipeline, depth=99)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)
    renderer = JSRenderer()
    rf_data = renderer.render(graph_data, theme="dark", separate_outputs=True, show_types=True)
    html = generate_widget_html(rf_data)
    
    # Save to file
    with open("outputs/test_overflow_fix.html", "w") as f:
        f.write(html)
    
    print("HTML saved to outputs/test_overflow_fix.html")
    print("\nNodes with type hints:")
    for n in rf_data["nodes"]:
        if n.get("data", {}).get("nodeType") == "DATA":
            label = n.get("data", {}).get("label", "")
            type_hint = n.get("data", {}).get("typeHint", "")
            if type_hint:
                print(f"  - {label} : {type_hint}")


if __name__ == "__main__":
    main()
