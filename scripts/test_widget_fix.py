"""
Test script to verify the widget fix works.
This can be run in a Jupyter notebook or VSCode notebook.
"""

from hypernodes import Pipeline, node
from hypernodes.viz.visualization_widget import PipelineWidget


@node(output_name="y")
def my_node(x: int) -> int:
    return x + 1


@node(output_name="z")
def another_node(y: int) -> int:
    return y * 2


# Create a simple pipeline
pipeline = Pipeline([my_node, another_node])

# Create the widget
print("Creating widget...")
widget = PipelineWidget(pipeline)

print(f"Widget type: {type(widget)}")
print(f"Widget value length: {len(widget.value)}")
print("\nWidget value preview (first 500 chars):")
print(widget.value[:500])

# Check if it uses srcdoc
if 'srcdoc=' in widget.value:
    print("\n✓ Using srcdoc attribute (good for VSCode compatibility)")
elif 'data:text/html;base64' in widget.value:
    print("\n✗ Still using base64 data URI (may not work in VSCode)")
else:
    print("\n? Unknown iframe method")

print("\n" + "="*60)
print("To test in a notebook:")
print("1. Copy the code below into a Jupyter/VSCode notebook cell")
print("2. Run the cell and check if the visualization appears")
print("="*60)
print("""
from hypernodes import Pipeline, node
from hypernodes.viz.visualization_widget import PipelineWidget

@node(output_name="y")
def my_node(x: int) -> int:
    return x + 1

pipeline = Pipeline([my_node])
widget = PipelineWidget(pipeline)
widget
""")

