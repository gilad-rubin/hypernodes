import uuid
import html
from typing import Any, Optional

import ipywidgets as widgets
from IPython.display import display

from ..pipeline import Pipeline
from ..ui_handler import UIHandler
from .renderer import GraphvizRenderer


class GraphvizWidget(widgets.VBox):
    """Interactive Graphviz widget for Jupyter notebooks."""

    def __init__(
        self,
        pipeline: Pipeline,
        theme: str = "default",
        depth: Optional[int] = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        
        self.pipeline = pipeline
        self.handler = UIHandler(pipeline, depth=depth)
        self.renderer = GraphvizRenderer(style=theme)
        self.uid = str(uuid.uuid4())[:8]
        
        # Communication channel (hidden input)
        # We use a Text widget that JS will write to
        self.comm = widgets.Text(value="", layout=widgets.Layout(display="none"))
        self.comm.add_class(f"hn-comm-{self.uid}")
        self.comm.observe(self._on_comm_message, names="value")
        
        # Display area for SVG
        self.html_out = widgets.HTML(
            value="Rendering...",
            layout=widgets.Layout(width="100%", overflow="auto")
        )
        
        self.children = [self.html_out, self.comm]
        
        # Initial render
        self._render()

    def _on_comm_message(self, change: Any) -> None:
        """Handle messages from JavaScript."""
        msg = change["new"]
        if not msg:
            return
            
        # Reset comm buffer to allow re-sending same message
        self.comm.value = ""
        
        try:
            if msg.startswith("expand:"):
                node_id = msg.split(":", 1)[1]
                self.handler.toggle_node(node_id)
                self._render()
        except Exception as e:
            self.html_out.value = f"<div style='color:red'>Error: {e}</div>"

    def _render(self) -> None:
        """Render the graph and update the HTML widget."""
        try:
            graph_data = self.handler.get_visualization_data()
            svg_content = self.renderer.render(graph_data)
            
            # Inject the interaction script
            # We find the specific comm input for this widget instance
            script = f"""
            <script>
            (function() {{
                // Find the comm input for this specific widget instance
                const findComm = () => document.querySelector('.hn-comm-{self.uid} input');
                
                // The SVG is rendered in the previous sibling's child (the HTML widget)
                // We attach the listener to the container of this script
                const container = document.currentScript.parentElement;
                
                container.addEventListener('click', function(e) {{
                    // Traverse up to find anchor tag
                    let target = e.target;
                    while (target && target.tagName !== 'A' && target !== container) {{
                        target = target.parentElement;
                    }}
                    
                    if (target && target.tagName === 'A') {{
                        // Check standard href and xlink:href (for SVG compatibility)
                        const href = target.getAttribute('href') || target.getAttribute('xlink:href');
                        
                        if (href && href.startsWith('hypernodes:expand')) {{
                            e.preventDefault();
                            e.stopPropagation();
                            
                            const urlParams = new URLSearchParams(href.split('?')[1]);
                            const nodeId = urlParams.get('id');
                            
                            const commInput = findComm();
                            if (commInput && nodeId) {{
                                commInput.value = 'expand:' + nodeId;
                                // Trigger input event for Traitlets observer
                                commInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            }}
                        }}
                    }}
                }});
            }})();
            </script>
            """
            
            self.html_out.value = f"<div>{svg_content}{script}</div>"
            
        except Exception as e:
            self.html_out.value = f"<div style='color:red'>Render Error: {e}</div>"

