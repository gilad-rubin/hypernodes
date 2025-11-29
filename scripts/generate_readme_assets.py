import graphviz
import os

def generate_run_vs_map():
    """Generates a comparison diagram for Run vs Map."""
    
    # Create the main graph
    dot = graphviz.Digraph(comment='Run vs Map', format='svg')
    dot.attr(rankdir='LR', compound='true', splines='ortho', nodesep='0.8', ranksep='1.0')
    dot.attr('node', shape='rect', style='rounded,filled', fillcolor='white', fontname='Arial', fontsize='12', height='0.6')
    dot.attr('edge', arrowsize='0.8', color='#555555')

    # --- Cluster: Pipeline.run() ---
    with dot.subgraph(name='cluster_run') as c:
        c.attr(label='Pipeline.run()', fontname='Arial', fontsize='14', style='dashed', color='#888888', fontcolor='#555555', margin='20')
        
        c.node('input_run', 'Input Item', shape='note', fillcolor='#E3F2FD', color='#2196F3')
        c.node('node1_run', 'Node A')
        c.node('node2_run', 'Node B')
        c.node('output_run', 'Output', shape='note', fillcolor='#E8F5E9', color='#4CAF50')

        c.edge('input_run', 'node1_run')
        c.edge('node1_run', 'node2_run')
        c.edge('node2_run', 'output_run')

    # --- Cluster: Pipeline.map() ---
    with dot.subgraph(name='cluster_map') as c:
        c.attr(label='Pipeline.map()', fontname='Arial', fontsize='14', style='dashed', color='#888888', fontcolor='#555555', margin='20')
        
        # Inputs
        c.node('input_map_1', 'Item 1', shape='note', fillcolor='#E3F2FD', color='#2196F3')
        c.node('input_map_2', 'Item 2', shape='note', fillcolor='#E3F2FD', color='#2196F3')
        c.node('input_map_3', 'Item 3', shape='note', fillcolor='#E3F2FD', color='#2196F3')

        # The "Pipeline" box
        with c.subgraph(name='cluster_pipeline') as p:
            p.attr(label='The Pipeline', style='filled', color='#F5F5F5', fillcolor='#F5F5F5')
            p.node('node1_map', 'Node A', fillcolor='white')
            p.node('node2_map', 'Node B', fillcolor='white')
            p.edge('node1_map', 'node2_map')

        # Outputs
        c.node('output_map_1', 'Result 1', shape='note', fillcolor='#E8F5E9', color='#4CAF50')
        c.node('output_map_2', 'Result 2', shape='note', fillcolor='#E8F5E9', color='#4CAF50')
        c.node('output_map_3', 'Result 3', shape='note', fillcolor='#E8F5E9', color='#4CAF50')

        # Edges showing the "fan out"
        # We use invisible nodes or ports to make it look like they go through the pipeline
        
        # To make it look like parallel execution, we can just draw lines through the pipeline box
        # But graphviz layout is tricky. 
        # Let's try a different approach: The pipeline is a "function" applied to each.
        
        c.edge('input_map_1', 'node1_map', lhead='cluster_pipeline', style='dashed', color='#2196F3')
        c.edge('input_map_2', 'node1_map', lhead='cluster_pipeline', style='dashed', color='#2196F3')
        c.edge('input_map_3', 'node1_map', lhead='cluster_pipeline', style='dashed', color='#2196F3')

        c.edge('node2_map', 'output_map_1', ltail='cluster_pipeline', style='dashed', color='#4CAF50')
        c.edge('node2_map', 'output_map_2', ltail='cluster_pipeline', style='dashed', color='#4CAF50')
        c.edge('node2_map', 'output_map_3', ltail='cluster_pipeline', style='dashed', color='#4CAF50')

    # Save
    output_path = 'assets/readme/run_vs_map'
    dot.render(output_path, cleanup=True)
    print(f"Generated {output_path}.svg")

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs('assets/readme', exist_ok=True)
    generate_run_vs_map()
