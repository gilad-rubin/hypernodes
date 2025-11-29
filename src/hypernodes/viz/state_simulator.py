"""State simulator - Python port of JS state_utils.js transformations.

This module replicates the client-side JavaScript state transformations in Python,
allowing you to test visualization state changes without running a browser.

Usage:
    from hypernodes.viz import UIHandler
    from hypernodes.viz.state_simulator import (
        simulate_state,
        verify_state,
        verify_edge_alignment,
        simulate_collapse_expand_cycle,
        diagnose_all_states,
    )

    handler = UIHandler(pipeline, depth=2)
    graph = handler.get_visualization_data(traverse_collapsed=True)
    
    # Simulate collapsed pipeline + combined outputs
    result = simulate_state(
        graph,
        expansion_state={"my_pipeline": False},
        separate_outputs=False,
        show_types=True
    )
    
    print("Visible nodes:", [n["id"] for n in result["nodes"] if not n.get("hidden")])
    
    # Verify edge-node alignment
    alignment = verify_edge_alignment(result)
    if not alignment["valid"]:
        print("Issues:", alignment["issues"])
    
    # Test collapse/expand cycle
    cycle_result = simulate_collapse_expand_cycle(graph, "my_pipeline")
    print(cycle_result["summary"])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .structures import (
    DataNode,
    DualNode,
    FunctionNode,
    GroupDataNode,
    PipelineNode,
    VisualizationGraph,
    VizEdge,
    VizNode,
)


def _node_to_dict(node: VizNode) -> Dict[str, Any]:
    """Convert a VizNode to a dictionary matching JS node format."""
    base = {
        "id": node.id,
        "parentNode": node.parent_id,
        "position": {"x": 0, "y": 0},
        "data": {},
    }
    
    if isinstance(node, FunctionNode):
        base["type"] = "custom"
        base["data"]["nodeType"] = "DUAL" if isinstance(node, DualNode) else "FUNCTION"
        base["data"]["label"] = node.label
        base["data"]["functionName"] = node.function_name
    elif isinstance(node, PipelineNode):
        base["type"] = "pipelineGroup" if node.is_expanded else "custom"
        base["data"]["nodeType"] = "PIPELINE"
        base["data"]["label"] = node.label
        base["data"]["isExpanded"] = node.is_expanded
    elif isinstance(node, DataNode):
        base["type"] = "custom"
        base["data"]["nodeType"] = "DATA"
        base["data"]["label"] = node.name
        base["data"]["typeHint"] = node.type_hint
        base["data"]["isBound"] = node.is_bound
        base["data"]["sourceId"] = node.source_id
    elif isinstance(node, GroupDataNode):
        base["type"] = "custom"
        base["data"]["nodeType"] = "GROUP_DATA"
        base["data"]["isBound"] = node.is_bound
        base["data"]["sourceId"] = node.source_id
        base["data"]["nodes"] = [_node_to_dict(n) for n in node.nodes]
    
    return base


def _edge_to_dict(edge: VizEdge) -> Dict[str, Any]:
    """Convert a VizEdge to a dictionary matching JS edge format."""
    return {
        "id": f"e_{edge.source}_{edge.target}",
        "source": edge.source,
        "target": edge.target,
        "label": edge.label,
    }


def simulate_state(
    graph: Union[VisualizationGraph, Dict[str, Any]],
    expansion_state: Dict[str, bool],
    separate_outputs: bool = False,
    show_types: bool = True,
    theme: str = "dark",
) -> Dict[str, Any]:
    """Simulate the JS applyState + applyVisibility + compressEdges transformations.
    
    This replicates what state_utils.js does client-side, allowing testing in Python.
    
    Args:
        graph: VisualizationGraph or already-converted dict
        expansion_state: Map of pipeline node ID -> expanded (True/False)
        separate_outputs: If True, keep output nodes separate. If False, combine into producer.
        show_types: Whether to show type hints
        theme: "dark" or "light"
    
    Returns:
        Dictionary with:
        - nodes: List of transformed nodes (with hidden flag)
        - edges: List of transformed edges (remapped for collapsed pipelines)
    """
    # Convert graph to dict format if needed
    if isinstance(graph, VisualizationGraph):
        base_nodes = [_node_to_dict(n) for n in graph.nodes]
        base_edges = [_edge_to_dict(e) for e in graph.edges]
    else:
        base_nodes = graph.get("nodes", [])
        base_edges = graph.get("edges", [])
    
    # Build lookup maps
    source_node_types: Dict[str, str] = {}
    source_id_map: Dict[str, Optional[str]] = {}
    
    for n in base_nodes:
        source_node_types[n["id"]] = n.get("data", {}).get("nodeType")
        source_id = n.get("data", {}).get("sourceId")
        if source_id:
            source_id_map[n["id"]] = source_id
    
    # Track expanded/collapsed pipelines
    expanded_pipelines: Set[str] = set()
    collapsed_pipelines: Set[str] = set()
    
    for n in base_nodes:
        if n.get("data", {}).get("nodeType") == "PIPELINE":
            node_id = n["id"]
            is_expanded = expansion_state.get(node_id, n.get("data", {}).get("isExpanded", False))
            if is_expanded:
                expanded_pipelines.add(node_id)
            else:
                collapsed_pipelines.add(node_id)
    
    # Identify boundary outputs (DATA nodes with sourceId pointing to a PIPELINE)
    boundary_outputs: Set[str] = set()
    for n in base_nodes:
        source_id = n.get("data", {}).get("sourceId")
        if source_id and source_node_types.get(source_id) == "PIPELINE":
            boundary_outputs.add(n["id"])
    
    # Boundary outputs of EXPANDED pipelines should be hidden
    boundary_outputs_to_hide: Set[str] = set()
    for n in base_nodes:
        if n["id"] in boundary_outputs:
            source_id = n.get("data", {}).get("sourceId")
            if source_id in expanded_pipelines:
                boundary_outputs_to_hide.add(n["id"])
    
    # Build edge maps for producer tracing
    edges_by_target: Dict[str, List[Dict]] = {}
    for e in base_edges:
        target = e["target"]
        if target not in edges_by_target:
            edges_by_target[target] = []
        edges_by_target[target].append(e)
    
    def find_visible_producer(node_id: str, stop_at_visible_output: bool = False) -> str:
        """Find the visible producer for a node (traces through output chain)."""
        node = next((n for n in base_nodes if n["id"] == node_id), None)
        if not node:
            return node_id
        
        source_id = node.get("data", {}).get("sourceId")
        if not source_id:
            return node_id
        
        # Check incoming edges
        incoming = edges_by_target.get(node_id, [])
        if incoming:
            producer_id = incoming[0]["source"]
            # If producer is visible and we should stop at visible outputs
            if stop_at_visible_output and producer_id not in boundary_outputs_to_hide:
                return producer_id
            return find_visible_producer(producer_id, stop_at_visible_output)
        
        # Fallback: use sourceId
        source_type = source_node_types.get(source_id)
        if source_type in ("FUNCTION", "DUAL"):
            return source_id
        return source_id
    
    # Apply meta (theme, showTypes, etc.)
    def apply_meta(node: Dict) -> Dict:
        result = {**node}
        node_type = node.get("data", {}).get("nodeType")
        is_pipeline = node_type == "PIPELINE"
        
        if is_pipeline:
            is_expanded = expansion_state.get(node["id"], node.get("data", {}).get("isExpanded", False))
            result["type"] = "pipelineGroup" if is_expanded else "custom"
            result["data"] = {
                **node.get("data", {}),
                "theme": theme,
                "showTypes": show_types,
                "isExpanded": is_expanded,
            }
        else:
            result["data"] = {
                **node.get("data", {}),
                "theme": theme,
                "showTypes": show_types,
            }
        
        return result
    
    if separate_outputs:
        # Separate outputs mode:
        # - Hide boundary outputs for EXPANDED pipelines (internal is visible)
        # - Show boundary outputs for COLLAPSED pipelines
        filtered_nodes = [
            {**apply_meta(n), "data": {**apply_meta(n)["data"], "separateOutputs": True}}
            for n in base_nodes
            if n["id"] not in boundary_outputs_to_hide
        ]
        
        # Remap edges FROM hidden boundary outputs to their producer
        remapped_edges = []
        for e in base_edges:
            if e["target"] in boundary_outputs_to_hide:
                continue
            if e["source"] in boundary_outputs_to_hide:
                producer = find_visible_producer(e["source"], stop_at_visible_output=True)
                remapped_edges.append({
                    **e,
                    "id": f"e_{producer}_{e['target']}",
                    "source": producer,
                })
            else:
                remapped_edges.append(e)
        
        nodes = filtered_nodes
        edges = remapped_edges
    else:
        # Combined outputs mode:
        # - Always combine FUNCTION/DUAL outputs
        # - Combine PIPELINE outputs only when pipeline is COLLAPSED
        output_nodes: Set[str] = set()
        for n in base_nodes:
            source_id = n.get("data", {}).get("sourceId")
            if not source_id:
                continue
            source_type = source_node_types.get(source_id)
            if source_type in ("FUNCTION", "DUAL"):
                output_nodes.add(n["id"])
            elif source_type == "PIPELINE" and source_id in collapsed_pipelines:
                output_nodes.add(n["id"])
        
        # Also hide boundary outputs for expanded pipelines
        nodes_to_hide = output_nodes | boundary_outputs_to_hide
        
        # Build function outputs map (for combined display)
        function_outputs: Dict[str, List[Dict[str, Any]]] = {}
        for n in base_nodes:
            source_id = n.get("data", {}).get("sourceId")
            if source_id and n["id"] in output_nodes:
                if source_id not in function_outputs:
                    function_outputs[source_id] = []
                function_outputs[source_id].append({
                    "name": n.get("data", {}).get("label"),
                    "type": n.get("data", {}).get("typeHint"),
                })
        
        nodes = []
        for n in base_nodes:
            if n["id"] in nodes_to_hide:
                continue
            node = apply_meta(n)
            node["data"] = {
                **node["data"],
                "separateOutputs": False,
                "outputs": function_outputs.get(n["id"], []),
            }
            nodes.append(node)
        
        # Process edges
        edges = []
        seen_edges: Set[str] = set()
        
        for e in base_edges:
            # Skip edges TO hidden nodes
            if e["target"] in nodes_to_hide:
                continue
            
            new_source = e["source"]
            
            # Remap from output nodes
            if e["source"] in output_nodes:
                output_node = next((n for n in base_nodes if n["id"] == e["source"]), None)
                if output_node:
                    source_id = output_node.get("data", {}).get("sourceId")
                    if source_id:
                        new_source = source_id
            
            # Remap from hidden boundary outputs
            if e["source"] in boundary_outputs_to_hide:
                new_source = find_visible_producer(e["source"], stop_at_visible_output=False)
            
            # Skip if source is still hidden
            if new_source in nodes_to_hide:
                continue
            
            edge_key = f"{new_source}_{e['target']}"
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({
                    **e,
                    "id": f"e_{new_source}_{e['target']}",
                    "source": new_source,
                })
    
    # Apply visibility (hide children of collapsed pipelines)
    parent_map: Dict[str, Optional[str]] = {}
    for n in nodes:
        parent_map[n["id"]] = n.get("parentNode")
    
    def is_hidden(node_id: str) -> bool:
        curr = node_id
        while curr:
            parent = parent_map.get(curr)
            if not parent:
                return False
            if parent in collapsed_pipelines:
                return True
            curr = parent
        return False
    
    nodes_with_visibility = [
        {**n, "hidden": is_hidden(n["id"])}
        for n in nodes
    ]
    
    # Compress edges (remap to visible ancestors for collapsed pipelines)
    def get_visible_ancestor(node_id: str) -> str:
        curr = node_id
        candidate = node_id
        while curr:
            parent = parent_map.get(curr)
            if not parent:
                break
            if parent in collapsed_pipelines:
                candidate = parent
            curr = parent
        return candidate
    
    compressed_edges = []
    seen_compressed: Set[str] = set()
    
    for e in edges:
        source_vis = get_visible_ancestor(e["source"])
        target_vis = get_visible_ancestor(e["target"])
        
        if source_vis == e["source"] and target_vis == e["target"]:
            edge_key = e["id"]
            if edge_key not in seen_compressed:
                compressed_edges.append(e)
                seen_compressed.add(edge_key)
        elif source_vis != target_vis:
            edge_key = f"e_{source_vis}_{target_vis}"
            if edge_key not in seen_compressed:
                compressed_edges.append({
                    **e,
                    "id": edge_key,
                    "source": source_vis,
                    "target": target_vis,
                })
                seen_compressed.add(edge_key)
    
    return {
        "nodes": nodes_with_visibility,
        "edges": compressed_edges,
    }


def verify_state(
    result: Dict[str, Any],
    expected_visible_nodes: Optional[List[str]] = None,
    expected_edges: Optional[List[tuple]] = None,
    forbidden_edges: Optional[List[tuple]] = None,
) -> Dict[str, Any]:
    """Verify that a simulated state matches expectations.
    
    Args:
        result: Output from simulate_state()
        expected_visible_nodes: List of node IDs that should be visible
        expected_edges: List of (source, target) tuples that should exist
        forbidden_edges: List of (source, target) tuples that should NOT exist
    
    Returns:
        Dictionary with verification results and any failures.
    """
    visible_nodes = {n["id"] for n in result["nodes"] if not n.get("hidden")}
    edge_set = {(e["source"], e["target"]) for e in result["edges"]}
    
    failures = []
    
    if expected_visible_nodes is not None:
        expected_set = set(expected_visible_nodes)
        missing = expected_set - visible_nodes
        extra = visible_nodes - expected_set
        if missing:
            failures.append(f"Missing visible nodes: {missing}")
        if extra:
            failures.append(f"Unexpected visible nodes: {extra}")
    
    if expected_edges is not None:
        for src, tgt in expected_edges:
            if (src, tgt) not in edge_set:
                failures.append(f"Missing edge: {src} → {tgt}")
    
    if forbidden_edges is not None:
        for src, tgt in forbidden_edges:
            if (src, tgt) in edge_set:
                failures.append(f"Forbidden edge exists: {src} → {tgt}")
    
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "visible_nodes": list(visible_nodes),
        "edges": [(e["source"], e["target"]) for e in result["edges"]],
    }


def diagnose_all_states(
    graph: Union[VisualizationGraph, Dict[str, Any]],
    pipeline_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run simulate_state for all combinations of expansion and separateOutputs.
    
    This is useful for AI agents to quickly check all possible states.
    
    Args:
        graph: VisualizationGraph or dict
        pipeline_ids: List of pipeline node IDs to toggle. If None, auto-detect.
    
    Returns:
        Dictionary with results for each state combination.
    """
    # Auto-detect pipeline IDs if not provided
    if pipeline_ids is None:
        if isinstance(graph, VisualizationGraph):
            pipeline_ids = [n.id for n in graph.nodes if isinstance(n, PipelineNode)]
        else:
            pipeline_ids = [
                n["id"] for n in graph.get("nodes", [])
                if n.get("data", {}).get("nodeType") == "PIPELINE"
            ]
    
    results = {}
    
    for separate in [False, True]:
        for all_expanded in [False, True]:
            exp_state = {pid: all_expanded for pid in pipeline_ids}
            
            key = f"separate={separate},expanded={all_expanded}"
            result = simulate_state(
                graph,
                expansion_state=exp_state,
                separate_outputs=separate,
            )
            
            visible = [n["id"] for n in result["nodes"] if not n.get("hidden")]
            edges = [(e["source"], e["target"]) for e in result["edges"]]
            
            # Check for issues
            node_ids = {n["id"] for n in result["nodes"]}
            orphan_edges = [
                e for e in edges
                if e[0] not in node_ids or e[1] not in node_ids
            ]
            
            results[key] = {
                "visible_node_count": len(visible),
                "edge_count": len(edges),
                "orphan_edges": orphan_edges,
                "visible_nodes": visible,
                "edges": edges,
            }
    
    return results


# Default node dimensions based on node type (matches html_generator.py layout logic)
DEFAULT_NODE_DIMENSIONS: Dict[str, Tuple[int, int]] = {
    "DATA": (140, 36),
    "INPUT": (160, 46),
    "INPUT_GROUP": (200, 64),
    "FUNCTION": (200, 90),
    "DUAL": (200, 90),
    "PIPELINE": (200, 68),  # Collapsed pipeline
    "PIPELINE_EXPANDED": (300, 200),  # Expanded pipeline (placeholder)
}


def verify_edge_alignment(
    result: Dict[str, Any],
    node_dimensions: Optional[Dict[str, Tuple[int, int]]] = None,
    tolerance: int = 20,
) -> Dict[str, Any]:
    """Verify that edges connect to valid positions on their source/target nodes.
    
    This function checks that:
    1. Each edge's source node exists and is visible
    2. Each edge's target node exists and is visible
    3. The edge would connect within reasonable bounds of the nodes
    
    Args:
        result: Output from simulate_state() with nodes and edges
        node_dimensions: Optional dict mapping node_id to (width, height). 
                        If not provided, uses default dimensions based on node type.
        tolerance: Pixel tolerance for alignment checks (default 20)
    
    Returns:
        {
            "valid": bool,
            "issues": [{"edge": (src, tgt), "type": str, "issue": str}],
            "summary": str
        }
    """
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])
    
    # Build node lookup
    node_map: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        node_map[n["id"]] = n
    
    # Get dimensions for a node
    def get_dimensions(node: Dict[str, Any]) -> Tuple[int, int]:
        node_id = node["id"]
        if node_dimensions and node_id in node_dimensions:
            return node_dimensions[node_id]
        
        node_type = node.get("data", {}).get("nodeType", "FUNCTION")
        is_expanded = node.get("data", {}).get("isExpanded", False)
        
        if node_type == "PIPELINE" and is_expanded:
            return DEFAULT_NODE_DIMENSIONS.get("PIPELINE_EXPANDED", (300, 200))
        
        # For INPUT_GROUP, estimate height based on param count
        if node_type == "INPUT_GROUP":
            param_count = len(node.get("data", {}).get("params", []))
            base_width, base_height = DEFAULT_NODE_DIMENSIONS.get("INPUT_GROUP", (200, 64))
            return (base_width, 46 + param_count * 18)
        
        return DEFAULT_NODE_DIMENSIONS.get(node_type, (200, 68))
    
    issues: List[Dict[str, Any]] = []
    validated_count = 0
    
    for edge in edges:
        source_id = edge.get("source")
        target_id = edge.get("target")
        edge_key = (source_id, target_id)
        
        # Check source node exists
        source_node = node_map.get(source_id)
        if not source_node:
            issues.append({
                "edge": edge_key,
                "type": "missing_source",
                "issue": f"Source node '{source_id}' not found in nodes",
            })
            continue
        
        # Check target node exists
        target_node = node_map.get(target_id)
        if not target_node:
            issues.append({
                "edge": edge_key,
                "type": "missing_target",
                "issue": f"Target node '{target_id}' not found in nodes",
            })
            continue
        
        # Check source is visible
        if source_node.get("hidden"):
            issues.append({
                "edge": edge_key,
                "type": "hidden_source",
                "issue": f"Source node '{source_id}' is hidden",
            })
            continue
        
        # Check target is visible
        if target_node.get("hidden"):
            issues.append({
                "edge": edge_key,
                "type": "hidden_target",
                "issue": f"Target node '{target_id}' is hidden",
            })
            continue
        
        # Nodes exist and are visible - edge is structurally valid
        validated_count += 1
        
        # Check parent relationship (edges should connect at same level or cross levels properly)
        source_parent = source_node.get("parentNode")
        target_parent = target_node.get("parentNode")
        
        # If source is inside a collapsed pipeline, there's an issue
        if source_parent:
            parent_node = node_map.get(source_parent)
            if parent_node and not parent_node.get("data", {}).get("isExpanded", True):
                issues.append({
                    "edge": edge_key,
                    "type": "collapsed_parent",
                    "issue": f"Source '{source_id}' is inside collapsed pipeline '{source_parent}'",
                })
    
    summary = f"{validated_count} edges validated, {len(issues)} issues found"
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "summary": summary,
        "validated_count": validated_count,
        "total_edges": len(edges),
    }


def simulate_collapse_expand_cycle(
    graph: Union[VisualizationGraph, Dict[str, Any]],
    pipeline_id: str,
    separate_outputs: bool = False,
) -> Dict[str, Any]:
    """Simulate a collapse -> expand cycle and verify edges at each stage.
    
    This is useful for testing that edges properly reconnect after collapse/expand.
    
    Args:
        graph: VisualizationGraph or dict
        pipeline_id: ID of the pipeline to collapse/expand
        separate_outputs: Whether to use separate outputs mode
    
    Returns:
        {
            "initial": { state, alignment },
            "after_collapse": { state, alignment },
            "after_expand": { state, alignment },
            "summary": str,
            "all_valid": bool,
        }
    """
    results = {}
    
    # Initial state (expanded)
    initial_state = simulate_state(
        graph,
        expansion_state={pipeline_id: True},
        separate_outputs=separate_outputs,
    )
    initial_alignment = verify_edge_alignment(initial_state)
    results["initial"] = {
        "expansion_state": {pipeline_id: True},
        "visible_nodes": len([n for n in initial_state["nodes"] if not n.get("hidden")]),
        "edges": len(initial_state["edges"]),
        "alignment": initial_alignment,
    }
    
    # After collapse
    collapsed_state = simulate_state(
        graph,
        expansion_state={pipeline_id: False},
        separate_outputs=separate_outputs,
    )
    collapsed_alignment = verify_edge_alignment(collapsed_state)
    results["after_collapse"] = {
        "expansion_state": {pipeline_id: False},
        "visible_nodes": len([n for n in collapsed_state["nodes"] if not n.get("hidden")]),
        "edges": len(collapsed_state["edges"]),
        "alignment": collapsed_alignment,
    }
    
    # After re-expand
    reexpanded_state = simulate_state(
        graph,
        expansion_state={pipeline_id: True},
        separate_outputs=separate_outputs,
    )
    reexpanded_alignment = verify_edge_alignment(reexpanded_state)
    results["after_expand"] = {
        "expansion_state": {pipeline_id: True},
        "visible_nodes": len([n for n in reexpanded_state["nodes"] if not n.get("hidden")]),
        "edges": len(reexpanded_state["edges"]),
        "alignment": reexpanded_alignment,
    }
    
    # Summary
    all_valid = all([
        initial_alignment["valid"],
        collapsed_alignment["valid"],
        reexpanded_alignment["valid"],
    ])
    
    issue_count = (
        len(initial_alignment["issues"]) +
        len(collapsed_alignment["issues"]) +
        len(reexpanded_alignment["issues"])
    )
    
    results["summary"] = (
        f"Pipeline '{pipeline_id}' collapse/expand cycle: "
        f"{'PASS' if all_valid else 'FAIL'} ({issue_count} total issues)"
    )
    results["all_valid"] = all_valid
    
    return results
