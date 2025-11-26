import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from typing import List
from hypernodes import Pipeline, node
from hypernodes.viz.graph_walker import GraphWalker
from hypernodes.viz.structures import DataNode

def test_pipeline_node_output_type():
    @node(output_name="result")
    def process(x: int) -> List[str]:
        return [str(x)]

    # Inner pipeline
    inner = Pipeline(nodes=[process])
    
    # Wrap as node WITH MAPPING
    inner_node = inner.as_node(name="inner_wrapper", output_mapping={"result": "final_result"})
    
    # Outer pipeline
    outer = Pipeline(nodes=[inner_node])
    
    # Generate viz data
    walker = GraphWalker(outer, expanded_nodes=set())
    viz_data = walker.get_visualization_data()
    
    # Find output node for "inner_wrapper"
    found = False
    for n in viz_data.nodes:
        if isinstance(n, DataNode) and n.source_id == "inner_wrapper":
            print(f"Found output node: {n.id}, name: {n.name}, type_hint: {n.type_hint}")
            if n.name == "final_result" and n.type_hint == "List[str]":
                found = True
            else:
                print(f"Expected name='final_result' and type='List[str]'")
    
    if found:
        print("SUCCESS: Pipeline mapped output type hint found!")
    else:
        print("FAILURE: Pipeline mapped output type hint NOT found.")
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline_node_output_type()

