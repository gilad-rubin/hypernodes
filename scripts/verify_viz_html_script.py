import json
import re
from hypernodes import Pipeline, node

# --- Setup Pipeline ---

@node(output_name="a_out")
def func_a(x: int) -> int:
    return x + 1

@node(output_name="b_out")
def func_b(y: int) -> int:
    return y * 2

@node(output_name="c_out")
def func_c(z: int) -> int:
    return z - 1

# Level 3
p3 = Pipeline(nodes=[func_c], name="p3")

# Level 2
n_p3 = p3.as_node(input_mapping={"b_out": "z"})
p2 = Pipeline(nodes=[func_b, n_p3], name="p2")

# Level 1 (Root)
n_p2 = p2.as_node(input_mapping={"a_out": "y"})
p1 = Pipeline(nodes=[func_a, n_p2], name="p1")

# --- Generate HTML ---

print("Generating HTML visualization...")
# We use engine="ipywidget" to trigger the full graph generation with state
# But since we are running in a script, we want to capture the HTML output.
# The visualize method with engine="ipywidget" returns a widget.
# However, for the purpose of checking the HTML generation logic, we can use the internal
# logic or just check if we can get the HTML string.

# Let's use the `visualize` method with return_type="html" but engine="ipywidget" 
# is usually for interactive use.
# The user asked "how do i run the html viz".
# Usually it's pipeline.visualize(engine="ipywidget") in a notebook.
# To verify it works, we can simulate what the widget does:
# It calls `transform_to_react_flow` with `handler.get_full_graph_with_state()`.

from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.visualization_widget import transform_to_react_flow, generate_widget_html

# 1. Test UIHandler integration
handler = UIHandler(p1)
handler.set_initial_depth(2) # p2 expanded, p3 collapsed
full_graph = handler.get_full_graph_with_state()

print(f"Full graph nodes: {len(full_graph['nodes'])}")

# Check is_expanded flags
p2_node = next(n for n in full_graph['nodes'] if n.get('label') == 'p2')
p3_node = next(n for n in full_graph['nodes'] if n.get('label') == 'p3')

print(f"p2 expanded: {p2_node.get('is_expanded')}")
print(f"p3 expanded: {p3_node.get('is_expanded')}")

if not p2_node.get('is_expanded'):
    print("ERROR: p2 should be expanded at depth 2")
    exit(1)
if p3_node.get('is_expanded'):
    print("ERROR: p3 should be collapsed at depth 2")
    exit(1)

# 2. Test HTML generation
react_flow_data = transform_to_react_flow(full_graph, theme="CYBERPUNK", initial_depth=2)
html_content = generate_widget_html(react_flow_data)

output_file = "viz_test.html"
with open(output_file, "w") as f:
    f.write(html_content)

print(f"HTML generated at {output_file}")

# Verify JSON in HTML
match = re.search(r'<script id="graph-data" type="application/json">(.*?)</script>', html_content, re.DOTALL)
if match:
    json_str = match.group(1)
    data = json.loads(json_str)
    print("Successfully extracted JSON from HTML")
    print(f"JSON nodes count: {len(data['nodes'])}")
    
    # verify p2 expansion in the react flow data
    # Note: transform_to_react_flow might rename IDs or structure, but let's check
    # In react flow data, we look for the node corresponding to p2
    # The ID in full_graph was p2_node['id']
    
    rf_p2 = next((n for n in data['nodes'] if n['id'] == p2_node['id']), None)
    if rf_p2:
        print(f"ReactFlow p2 data: {rf_p2.get('data', {})}")
        if rf_p2['data'].get('isExpanded'):
             print("SUCCESS: p2 is expanded in generated HTML JSON")
        else:
             print("FAILURE: p2 is NOT expanded in generated HTML JSON")
             exit(1)
    else:
        print("WARNING: Could not find p2 node in ReactFlow data (ID mismatch?)")

else:
    print("ERROR: Could not find graph-data script tag in HTML")
    exit(1)
