
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from hypernodes.viz.visualization_widget import PipelineWidget
from hypernodes.pipeline import Pipeline

def test_widget_instantiation():
    print("Testing PipelineWidget instantiation...")
    
    # Create a dummy pipeline
    p = Pipeline(nodes=[])
    
    # Add some dummy nodes (mocking structure since we don't have full hypernodes env setup in this script easily without defining nodes)
    # Actually, let's use a real pipeline if possible, or just mock the object structure expected by UIHandler
    
    # Since UIHandler expects a pipeline object, let's define a simple one
    # We need to import Node? No, let's just trust the Pipeline object creation
    
    try:
        widget = PipelineWidget(p)
        print("Widget instantiated successfully.")
        
        # Check if height is in the HTML
        html_content = widget.value
        if 'height="500"' in html_content or 'height: 500px' in html_content:
             print("SUCCESS: Default height 500px found (for empty/small graph).")
        else:
             print(f"WARNING: Expected height 500px not found. Content snippet: {html_content[:200]}...")
             
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_widget_instantiation()
