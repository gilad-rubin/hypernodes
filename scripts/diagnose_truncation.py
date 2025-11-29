"""Diagnostic script to find truncation bugs in visualization nodes."""

from typing import List

from hypernodes import Pipeline, node
from hypernodes.viz import UIHandler
from hypernodes.viz.js.html_generator import generate_widget_html
from hypernodes.viz.js.renderer import JSRenderer


class VectorStore:
    pass


# Create nodes with various input/output name lengths and type hints
@node(output_name="result")
def short_inputs(x: int, y: str) -> int:
    """Very short input names."""
    return 0


@node(output_name="output")
def medium_inputs(query: str, context: List[str]) -> str:
    """Medium input names."""
    return ""


@node(output_name="store")
def long_type_inputs(vector_store: VectorStore, embedding_model: str) -> str:
    """Longer input names with custom types."""
    return ""


@node(output_name="data")
def single_char(a: int) -> int:
    """Single char input."""
    return 0


def main():
    # Create pipeline
    pipeline = Pipeline(nodes=[short_inputs, medium_inputs, long_type_inputs, single_char])
    
    # Generate visualization with show_types=True
    handler = UIHandler(pipeline, depth=99)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)
    renderer = JSRenderer()
    rf_data = renderer.render(graph_data, theme="dark", separate_outputs=True, show_types=True)
    html = generate_widget_html(rf_data)
    
    # Save HTML
    with open("outputs/diagnose_truncation.html", "w") as f:
        f.write(html)
    
    print("HTML saved to outputs/diagnose_truncation.html")
    
    # Print all nodes with their data for diagnosis
    print("\n=== NODE DATA ===")
    for n in rf_data["nodes"]:
        node_type = n.get("data", {}).get("nodeType", "")
        node_id = n.get("id", "")
        label = n.get("data", {}).get("label", "")
        type_hint = n.get("data", {}).get("typeHint", "")
        params = n.get("data", {}).get("params", [])
        param_types = n.get("data", {}).get("paramTypes", [])
        show_types = n.get("data", {}).get("showTypes", False)
        
        if node_type in ["INPUT", "INPUT_GROUP", "DATA"]:
            print(f"\n{node_type}: {node_id}")
            print(f"  label: '{label}'")
            print(f"  typeHint: '{type_hint}'")
            print(f"  showTypes: {show_types}")
            if params:
                print(f"  params: {params}")
                print(f"  paramTypes: {param_types}")
            
            # Calculate expected display
            if node_type == "INPUT":
                display = label
                if show_types and type_hint:
                    truncated_type = type_hint[:25] + "..." if len(type_hint) > 25 else type_hint
                    display += f" : {truncated_type}"
                print(f"  expected display: '{display}'")
                print(f"  display length: {len(display)} chars")
            elif node_type == "INPUT_GROUP":
                for i, p in enumerate(params):
                    t = param_types[i] if i < len(param_types) else ""
                    display = p
                    if show_types and t:
                        truncated_type = t[:25] + "..." if len(t) > 25 else t
                        display += f" : {truncated_type}"
                    print(f"  param[{i}] display: '{display}' ({len(display)} chars)")


if __name__ == "__main__":
    main()
