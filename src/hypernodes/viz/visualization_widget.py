import html
from typing import Any, Dict, Optional

import ipywidgets as widgets

from .js.html_generator import generate_widget_html
from .js.renderer import JSRenderer
from .ui_handler import UIHandler
from .layout_estimator import LayoutEstimator


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
        show_types: bool = True,
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
        iframe_html = (
            f"{css_fix}"
            f'<iframe srcdoc="{escaped_html}" '
            f'width="{est_width}" height="{est_height}" frameborder="0" '
            f'style="border: none; width: 100%; min-width: {est_width}px; height: {est_height}px; display: block; background: transparent;" '
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
