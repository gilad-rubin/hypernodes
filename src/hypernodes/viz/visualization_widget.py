import html
import uuid
from typing import Any, Dict, Optional

import ipywidgets as widgets

from .js.html_generator import generate_widget_html
from .js.renderer import JSRenderer
from .layout_estimator import LayoutEstimator
from .ui_handler import UIHandler


class PipelineWidget(widgets.HTML):
    """
    Widget for visualizing pipelines inside Jupyter/VS Code notebooks.
    """

    def __init__(
        self,
        pipeline: Any,
        theme: str = "auto",
        depth: Optional[int] = 1,
        group_inputs: bool = True,
        show_types: bool = False,
        separate_outputs: bool = False,
        theme_debug: bool = False,
        **kwargs: Any,
    ):
        self.pipeline = pipeline
        self.theme = theme
        self.depth = depth
        self.theme_debug = theme_debug
        
        # 1. Get Graph Data
        handler = UIHandler(
            self.pipeline,
            depth=depth,
            group_inputs=group_inputs,
            show_output_types=show_types,
        )
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        
        # 2. Estimate Layout Dimensions (Python-side)
        # We do this BEFORE React Flow transform to use the rich structure
        estimator = LayoutEstimator(graph_data)
        est_width, est_height = estimator.estimate()
        
        # Enforce minimums
        est_height = max(400, est_height)
        est_width = max(600, est_width)
        
        # 3. Transform to React Flow
        renderer = JSRenderer()
        react_flow_data = renderer.render(
            graph_data,
            theme=theme,
            initial_depth=depth or 1,
            theme_debug=theme_debug,
            pan_on_scroll=False, # Force disable scroll hijacking
            separate_outputs=separate_outputs,
            show_types=show_types,
        )
        
        # 4. Generate HTML
        html_content = generate_widget_html(react_flow_data)

        # Use srcdoc for better compatibility (VS Code, etc.)
        escaped_html = html.escape(html_content, quote=True)

        # CSS fix for VS Code white background on ipywidgets
        css_fix = """
        <style>
        .cell-output-ipywidget-background {
           background-color: transparent !important;
        }
        .jp-OutputArea-output {
           background-color: transparent;
        }
        </style>
        """

        # We set the iframe size to the estimated size
        # This ensures the notebook cell expands to fit the graph
        # Width is set to the estimated width to avoid excessive padding for narrow graphs
        iframe_html = (
            f"{css_fix}"
            f'<iframe srcdoc="{escaped_html}" '
            f'width="{est_width}" height="{est_height}" frameborder="0" '
            f'style="border: none; width: {est_width}px; max-width: 100%; height: {est_height}px; display: block; background: transparent; margin: 0 auto;" '
            f'sandbox="allow-scripts allow-same-origin allow-popups allow-forms">'
            f"</iframe>"
        )
        super().__init__(value=iframe_html, **kwargs)

    def _repr_html_(self) -> str:
        """Fallback for environments that prefer raw HTML over widgets."""
        return self.value

def transform_to_react_flow(
    graph_data: Any,
    theme: str = "CYBERPUNK",
    initial_depth: int = 1,
    theme_debug: bool = False,
) -> Dict[str, Any]:
    """Transform graph data to React Flow format (helper)."""
    renderer = JSRenderer()
    return renderer.render(
        graph_data,
        theme=theme,
        initial_depth=initial_depth,
        theme_debug=theme_debug,
    )


class ScrollablePipelineWidget(widgets.HTML):
    """
    Widget for visualizing pipelines that allows notebook scrolling.
    
    Uses an overlay approach: by default, scroll events pass through to the notebook.
    Click on the visualization to enable interaction mode (pan/zoom).
    Move mouse away from the widget to re-enable scrolling.
    
    The widget automatically resizes when content changes (e.g., expanding outputs).
    
    This solves the iframe scroll-blocking issue in VS Code notebooks.
    
    NOTE: For best results, use the display() method instead of just returning the widget:
        widget = ScrollablePipelineWidget(pipeline)
        widget.display()  # <-- Use this for click-to-interact to work
    """

    def __init__(
        self,
        pipeline: Any,
        theme: str = "auto",
        depth: Optional[int] = 1,
        group_inputs: bool = True,
        show_types: bool = False,
        separate_outputs: bool = False,
        theme_debug: bool = False,
        **kwargs: Any,
    ):
        """
        Args:
            pipeline: The pipeline to visualize.
            theme: Color theme ("auto", "light", "dark").
            depth: Initial expansion depth for nested pipelines.
            group_inputs: Whether to group input nodes.
            show_types: Whether to show type hints.
            separate_outputs: Whether to show outputs as separate nodes.
            theme_debug: Show theme detection debug info.
        """
        self.pipeline = pipeline
        self.theme = theme
        self.depth = depth
        self.theme_debug = theme_debug
        self.group_inputs = group_inputs
        self.show_types = show_types
        self.separate_outputs = separate_outputs
        
        # Generate unique IDs for this widget instance
        self._unique_id = uuid.uuid4().hex[:8]
        
        # Pre-generate the HTML
        self._html_content = self._generate_html()
        
        super().__init__(value=self._html_content, **kwargs)
    
    def _generate_html(self) -> str:
        """Generate the HTML with overlay for scroll-through and dynamic resizing."""
        wrapper_id = f"scroll-wrapper-{self._unique_id}"
        iframe_id = f"viz-iframe-{self._unique_id}"
        overlay_id = f"scroll-overlay-{self._unique_id}"
        
        # 1. Get Graph Data
        handler = UIHandler(
            self.pipeline,
            depth=self.depth,
            group_inputs=self.group_inputs,
            show_output_types=self.show_types,
        )
        graph_data = handler.get_visualization_data(traverse_collapsed=True)
        
        # 2. Estimate Layout Dimensions
        estimator = LayoutEstimator(graph_data)
        est_width, est_height = estimator.estimate()
        
        # Enforce minimums
        est_height = max(400, est_height)
        est_width = max(600, est_width)
        
        # 3. Transform to React Flow
        renderer = JSRenderer()
        react_flow_data = renderer.render(
            graph_data,
            theme=self.theme,
            initial_depth=self.depth or 1,
            theme_debug=self.theme_debug,
            pan_on_scroll=False,
            separate_outputs=self.separate_outputs,
            show_types=self.show_types,
        )
        
        # 4. Generate HTML
        html_content = generate_widget_html(react_flow_data)
        escaped_html = html.escape(html_content, quote=True)

        # CSS fix for VS Code white background on ipywidgets
        css_fix = """
        <style>
        .cell-output-ipywidget-background {
           background-color: transparent !important;
        }
        .jp-OutputArea-output {
           background-color: transparent;
        }
        </style>
        """
        
        return f'''
{css_fix}
<div id="{wrapper_id}" style="position: relative; width: {est_width}px; max-width: 100%; height: {est_height}px; margin: 0 auto; transition: height 0.3s ease-out;">
    <iframe 
        id="{iframe_id}"
        srcdoc="{escaped_html}" 
        width="{est_width}" 
        height="{est_height}" 
        frameborder="0" 
        style="border: none; display: block; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: transparent; transition: height 0.3s ease-out;"
        sandbox="allow-scripts allow-same-origin allow-popups allow-forms">
    </iframe>
    <div 
        id="{overlay_id}"
        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 10; cursor: default; transition: height 0.3s ease-out;"
    ></div>
</div>
<script>
(function() {{
    var wrapper = document.getElementById('{wrapper_id}');
    var overlay = document.getElementById('{overlay_id}');
    var iframe = document.getElementById('{iframe_id}');
    
    // Disable overlay interaction entirely to allow direct access to the widget
    overlay.style.pointerEvents = 'none';
    
    // Listen for messages from the iframe
    window.addEventListener('message', function(event) {{
        if (event.data && event.data.type === 'hypernodes-viz-resize') {{
            var newHeight = event.data.height;
            var newWidth = event.data.width;
            
            if (newHeight && wrapper) {{
                wrapper.style.height = newHeight + 'px';
                iframe.style.height = newHeight + 'px';
                iframe.height = newHeight;
            }}
            if (newWidth && wrapper) {{
                var currentWidth = parseInt(wrapper.style.width) || 0;
                if (newWidth > currentWidth) {{
                    wrapper.style.width = Math.min(newWidth, window.innerWidth * 0.95) + 'px';
                    iframe.style.width = '100%';
                }}
            }}
        }}
    }});
}})();
</script>
<p style="font-size: 12px; color: gray; margin-top: 5px;">Click to interact. Scrolling resumes automatically.</p>
'''

    def display(self):
        """Display the widget using IPython.display.HTML for proper script execution."""
        from IPython.display import HTML
        from IPython.display import display as ipy_display
        ipy_display(HTML(self._html_content))

    def _repr_html_(self) -> str:
        """Fallback for environments that prefer raw HTML over widgets."""
        return self._html_content
