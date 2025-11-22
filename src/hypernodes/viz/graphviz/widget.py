    def _render(self) -> None:
        """Render the graph and update the HTML widget."""
        try:
            # For Graphviz (static), don't traverse collapsed pipelines
            # traverse_collapsed=True is for interactive widgets only
            graph_data = self.handler.get_visualization_data(traverse_collapsed=False)
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
                
                // Add mouseover/mouseout listeners to the container
                container.addEventListener('mouseenter', () => {{
                    // When mouse is over the graph, we want scrolling to pan/zoom IF enabled
                    // But user requested scrolling to scroll the notebook.
                    // Graphviz SVG is wrapped in a div with overflow:auto, so it scrolls internally if content overflows.
                    // This is the default browser behavior. 
                }});
                
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
