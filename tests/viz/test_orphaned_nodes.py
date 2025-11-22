
from hypernodes import Pipeline, node
from hypernodes.viz.ui_handler import UIHandler


def test_orphaned_inputs_in_nested_pipeline():
    @node(output_name="clean")
    def clean_text(raw_text: str, suffix: str = "", language: str = "en") -> str:
        return f"{language}:{raw_text.strip()}{suffix}"

    inner = Pipeline(
        nodes=[clean_text],
        name="InnerPipeline",
    ).bind(suffix="*", language="en")

    inner_node = inner.as_node(
        name="InnerPipeline",
        input_mapping={"text_step": "raw_text"},
    )

    @node(output_name="text_step")
    def add_bang(text: str) -> str:
        return text + "!"

    outer = Pipeline(
        nodes=[add_bang, inner_node],
        name="OuterPipeline",
    )

    # Case 1: Collapsed (depth=1)
    # Inputs 'suffix' and 'language' are bound in InnerPipeline.
    # They should be associated with InnerPipeline node.
    handler = UIHandler(outer, depth=1)
    view = handler.get_view_data()
    
    # Check where input nodes for suffix/language are
    input_nodes = [n for n in view["nodes"] if n["node_type"] in ("INPUT", "INPUT_GROUP")]
    
    # In the buggy version, they might be at "root" level but visually disconnected or weirdly placed
    # The issue described is "orphan language, suffix node that goes into a white node"
    
    # Let's check the level_id of these inputs.
    # If InnerPipeline is collapsed, it is a node in 'root' level.
    # The inputs should be at 'root' level too if they feed into InnerPipeline.
    
    # However, if they are bound inputs of the inner pipeline, they might be grouped.
    
    # Let's look at the structure.
    # InnerPipeline node id in view
    inner_pipeline_node = next(n for n in view["nodes"] if n["label"] == "InnerPipeline")
    inner_id = inner_pipeline_node["id"]
    
    # Check for grouped inputs
    grouped_inputs = view.get("grouped_inputs", {})
    
    # If they are grouped, they should be associated with inner_id
    if inner_id in grouped_inputs:
        groups = grouped_inputs[inner_id]
        bound_params = groups.get("bound", [])
        assert "suffix" in bound_params
        assert "language" in bound_params
    else:
        # If not grouped, they should be individual INPUT nodes
        # But wait, the user says "orphan ... node".
        pass

    # Case 2: Expanded (depth=2)
    # InnerPipeline is expanded. It has a nested level id.
    # The inputs 'suffix' and 'language' should be inside that nested level.
    handler_expanded = UIHandler(outer, depth=2)
    view_expanded = handler_expanded.get_view_data()
    
    inner_pipeline_node_expanded = next(n for n in view_expanded["nodes"] if n["label"] == "InnerPipeline")
    # When expanded, the PipelineNode itself is still there but is_expanded=True
    # And there are children nodes.
    
    # Find the level of the inner pipeline content
    # The InnerPipeline node has a 'nested_level_id' in the full graph, but in view_data it might not be explicit
    # We can look at the nodes that are children of InnerPipeline.
    
    # clean_text node should be visible
    clean_text_node = next(n for n in view_expanded["nodes"] if n["label"] == "clean_text")
    inner_level_id = clean_text_node["level_id"]
    assert inner_level_id != "root"
    
    # The inputs 'suffix' and 'language' should be at inner_level_id
    # OR they should be grouped inputs for clean_text at inner_level_id
    
    # Check if they appear as INPUT nodes at root level (which would be wrong)
    root_inputs = [n for n in view_expanded["nodes"] 
                  if n["node_type"] == "INPUT" and n["level_id"] == "root"]
    root_input_labels = [n["label"] for n in root_inputs]
    
    assert "suffix" not in root_input_labels, "suffix should not be at root level when expanded"
    assert "language" not in root_input_labels, "language should not be at root level when expanded"
    
    # They should be at inner_level_id
    inner_inputs = [n for n in view_expanded["nodes"] 
                   if n["node_type"] == "INPUT" and n["level_id"] == inner_level_id]
    inner_input_labels = [n["label"] for n in inner_inputs]
    
    # Or they might be grouped inputs for clean_text
    clean_text_id = clean_text_node["id"]
    grouped_inputs_exp = view_expanded.get("grouped_inputs", {})
    
    found_suffix = False
    found_language = False
    
    if clean_text_id in grouped_inputs_exp:
        bound = grouped_inputs_exp[clean_text_id].get("bound", [])
        if "suffix" in bound:
            found_suffix = True
        if "language" in bound:
            found_language = True
        
    if not found_suffix:
        assert "suffix" in inner_input_labels, "suffix should be at inner level"
    if not found_language:
        assert "language" in inner_input_labels, "language should be at inner level"

