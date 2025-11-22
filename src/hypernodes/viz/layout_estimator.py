from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from .structures import VisualizationGraph, PipelineNode, FunctionNode, DataNode, GroupDataNode, VizNode

class LayoutEstimator:
    """
    Estimates the dimensions of the visualization graph to size the container appropriately.
    Mimics a simplified layered layout algorithm.
    """
    
    # Constants matching ELK/Renderer config
    NODE_WIDTH = 240
    NODE_HEIGHT = 80
    DATA_NODE_WIDTH = 120
    DATA_NODE_HEIGHT = 40
    INPUT_GROUP_WIDTH = 160
    INPUT_GROUP_BASE_HEIGHT = 40
    INPUT_ITEM_HEIGHT = 20
    
    SPACING_X = 60
    SPACING_Y = 80
    PADDING = 40
    
    GROUP_PADDING = 60  # Padding inside a group
    
    def __init__(self, graph: VisualizationGraph):
        self.graph = graph
        self.node_map = {n.id: n for n in graph.nodes}
        self.children_map = defaultdict(list)
        for n in graph.nodes:
            self.children_map[n.parent_id].append(n)
            
        # Filter edges to only those relevant for layout (structural dependencies)
        # We build an adjacency list for the whole graph, but will filter by scope
        self.adj = defaultdict(list)
        for e in graph.edges:
            self.adj[e.source].append(e.target)

    def estimate(self) -> Tuple[int, int]:
        """Returns (width, height) estimation for the root graph."""
        return self._estimate_scope(None)

    def _get_node_size(self, node: VizNode) -> Tuple[int, int]:
        if isinstance(node, PipelineNode) and node.is_expanded:
            # Recursive size for expanded pipeline
            w, h = self._estimate_scope(node.id)
            return w + self.GROUP_PADDING * 2, h + self.GROUP_PADDING * 2
        
        if isinstance(node, DataNode):
            width = self.DATA_NODE_WIDTH
            # Approximate width based on label length
            if hasattr(node, 'name') and node.name:
                 width = max(100, len(node.name) * 8 + 40)
            return width, self.DATA_NODE_HEIGHT
            
        if isinstance(node, GroupDataNode):
            count = len(node.nodes) if node.nodes else 1
            h = self.INPUT_GROUP_BASE_HEIGHT + (count * self.INPUT_ITEM_HEIGHT)
            return self.INPUT_GROUP_WIDTH, h
            
        # FunctionNode, PipelineNode (collapsed), etc.
        return self.NODE_WIDTH, self.NODE_HEIGHT

    def _estimate_scope(self, parent_id: Optional[str]) -> Tuple[int, int]:
        nodes = self.children_map[parent_id]
        if not nodes:
            return 0, 0
            
        # 1. Calculate size of every node in this scope
        node_sizes = {n.id: self._get_node_size(n) for n in nodes}
        
        # 2. Build local dependency graph for layering
        # We only consider edges where both source and target are in this scope
        # OR edges that connect to/from these nodes (but strict layering only works if contained)
        # Simplification: Just use index-based or simple topological sort approximation
        
        node_ids = set(n.id for n in nodes)
        local_adj = defaultdict(list)
        in_degree = {n.id: 0 for n in nodes}
        
        for u in node_ids:
            for v in self.adj[u]:
                if v in node_ids:
                    local_adj[u].append(v)
                    in_degree[v] += 1
                    
        # 3. Assign levels (Longest path layering)
        levels = defaultdict(list)
        node_level = {}
        
        queue = [n.id for n in nodes if in_degree[n.id] == 0]
        
        # If cycle or complex, just default to 0
        # We iterate to assign levels
        visited = set()
        
        # processing order
        processing_queue = [(nid, 0) for nid in queue]
        
        # Handle disconnected components / nodes with non-zero in-degree but no local inputs
        # (e.g. inputs from outside scope). Treat them as level 0.
        all_nodes_set = set(n.id for n in nodes)
        started_nodes = set(nid for nid, _ in processing_queue)
        remaining = all_nodes_set - started_nodes
        for r in remaining:
            # Heuristic: if not visited, put in level 0 (or handled later?)
            # Actually, if strict DAG, they should be reachable. If not, they are sources.
            # But `in_degree` check should have caught them if they have no incoming edges *within scope*.
            # If they have incoming edges from *outside*, they are sources in this scope.
            if in_degree[r] == 0:
                processing_queue.append((r, 0))
        
        final_levels = defaultdict(int)
        
        # Simple topological traversal to find max depth
        while processing_queue:
            u, lvl = processing_queue.pop(0)
            if u in visited:
                continue
            visited.add(u)
            
            final_levels[u] = max(final_levels[u], lvl)
            levels[final_levels[u]].append(u)
            
            for v in local_adj[u]:
                processing_queue.append((v, lvl + 1))
                
        # Handle any unvisited nodes (cycles?) - put them at level 0 or last?
        # Just append to level 0 for safety
        for n in nodes:
            if n.id not in visited:
                levels[0].append(n.id)
                
        # 4. Calculate Dimensions
        if not levels:
            return 0, 0
            
        num_levels = max(levels.keys()) + 1
        
        # Width of scope = Max width of any level
        max_level_width = 0
        
        # Height of scope = Sum of max height of each level
        total_height = 0
        
        for lvl in range(num_levels):
            lvl_nodes = [node_ids for node_ids in levels[lvl]]
            if not lvl_nodes: continue
            
            # Width of this level
            level_width = sum(node_sizes[nid][0] for nid in lvl_nodes) + (len(lvl_nodes) - 1) * self.SPACING_X
            max_level_width = max(max_level_width, level_width)
            
            # Height of this level
            level_height = max(node_sizes[nid][1] for nid in lvl_nodes)
            total_height += level_height
            
        # Add vertical spacing
        total_height += (num_levels - 1) * self.SPACING_Y
        
        # Add Padding
        return max_level_width + self.PADDING * 2, total_height + self.PADDING * 2

