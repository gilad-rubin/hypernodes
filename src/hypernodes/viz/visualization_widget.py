import html
from typing import Any, Dict, Optional

import ipywidgets as widgets

from .js.html_generator import generate_widget_html
from .js.renderer import JSRenderer
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
        
        # 2. Transform to React Flow
        renderer = JSRenderer()
        react_flow_data = renderer.render(
            graph_data,
            theme=theme,
            initial_depth=depth or 1,
            theme_debug=theme_debug,
        )
        
        # 3. Calculate Height
        estimated_height = self._calculate_initial_height(react_flow_data)
        
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

        iframe_html = (
            f"{css_fix}"
            f'<iframe srcdoc="{escaped_html}" '
            f'width="100%" height="{estimated_height}" frameborder="0" '
            f'style="border: none; width: 95%; max-width: 1600px; margin: 0; height: {estimated_height}px; display: block; background: transparent;" '
            f'sandbox="allow-scripts allow-same-origin allow-popups allow-forms">'
            f"</iframe>"
        )
        super().__init__(value=iframe_html, **kwargs)

    def _calculate_initial_height(self, graph: Dict[str, Any]) -> int:
        """Estimate the required height based on graph structure."""
        try:
            nodes = graph.get("nodes", [])
            num_nodes = len(nodes)
            
            # Linear scaling: Base 800 + 100px per node, capped at 3000
            calculated = 800 + (num_nodes * 100)
            return min(3000, calculated)
                
        except Exception:
            return 600

    def _repr_html_(self) -> str:
        """Fallback for environments that prefer raw HTML over widgets."""
        return self.value
