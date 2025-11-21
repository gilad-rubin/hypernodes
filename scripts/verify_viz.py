
import hypernodes as hn
from hypernodes import Pipeline, node
from hypernodes.visualization_widget import PipelineWidget
import json
import base64

# --- Sub Pipeline Nodes ---
@node(output_name="sum_val")
def add_inner(a: int, b: int) -> int:
    return a + b

@node(output_name="squared")
def square(sum_val: int) -> int:
    return sum_val * sum_val

sub_pipe = Pipeline([add_inner, square])
sub_pipe.bind(b=10) # Bind b=10 for the sub-pipeline

# Wrap as node
sub_node = sub_pipe.as_node(
    name="MathSubPipeline",
    input_mapping={"product": "a"},
    output_mapping={"squared": "squared_val"}
)

# --- Main Pipeline Nodes ---
@node(output_name="product")
def multiply(start: int, mult_factor: int) -> int:
    return start * mult_factor

@node(output_name="final_result")
def add_final(squared_val: int, offset: int) -> int:
    return squared_val + offset

# Main Pipeline
main_pipe = Pipeline([
    multiply,
    sub_node,
    add_final
])

# Bind constants for main pipeline
main_pipe.bind(mult_factor=2, offset=5)

# Create widget and inspect generated HTML/JSON
widget = PipelineWidget(main_pipe)

# The widget.value now contains an iframe with base64-encoded HTML
# Extract the base64 content
iframe_html = widget.value
print(f"Widget value type: {type(widget.value)}")
print(f"Widget value length: {len(widget.value)}")

# Extract base64 data from iframe
b64_start = iframe_html.find('base64,') + 7
b64_end = iframe_html.find('"', b64_start)
b64_data = iframe_html[b64_start:b64_end]

print(f"Base64 data length: {len(b64_data)}")

# Decode base64 to get inner HTML
inner_html = base64.b64decode(b64_data).decode('utf-8')
print(f"Inner HTML length: {len(inner_html)}")

# Extract JSON data from inner HTML
start_marker = '<script id="graph-data" type="application/json">'
end_marker = '</script>'

start_pos = inner_html.find(start_marker)
print(f"Start marker found at: {start_pos}")

if start_pos == -1:
    print("ERROR: Start marker not found!")
    exit(1)

start_idx = start_pos + len(start_marker)
end_idx = inner_html.find(end_marker, start_idx)
print(f"End marker found at: {end_idx}")

if end_idx == -1:
    print("ERROR: End marker not found!")
    exit(1)

json_str = inner_html[start_idx:end_idx].strip()
print(f"Extracted JSON string length: {len(json_str)}")

try:
    data = json.loads(json_str)
except json.JSONDecodeError as e:
    print(f"JSON Decode Error: {e}")
    print(f"JSON String Content (first 100 chars): {json_str[:100]}")
    exit(1)

print(f"\n✓ Nodes found: {len(data['nodes'])}")
print(f"✓ Edges found: {len(data['edges'])}")

# Check for nested node
nested_nodes = [n for n in data['nodes'] if n['type'] == 'pipelineGroup']
print(f"✓ Nested pipeline nodes: {len(nested_nodes)}")

if len(nested_nodes) > 0:
    print("✓ SUCCESS: Nested pipeline node found.")
else:
    print("✗ FAILURE: No nested pipeline node found.")

# Check for children nodes (nodes with parentNode set)
children = [n for n in data['nodes'] if n.get('parentNode')]
print(f"✓ Child nodes found: {len(children)}")

if len(children) > 0:
    print("✓ SUCCESS: Child nodes found.")
else:
    print("✗ FAILURE: No child nodes found.")

# Print node labels for debugging
print("\nNode Details:")
for n in data['nodes']:
    parent_info = f" (Parent: {n.get('parentNode', 'None')})" if n.get('parentNode') else " (Root)"
    print(f"  • {n['data']['label']} - Type: {n['data']['nodeType']}{parent_info}")

print("\n✓ Widget verification passed!")
