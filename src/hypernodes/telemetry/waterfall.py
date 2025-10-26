"""Waterfall chart visualization for execution analysis (Jupyter only)."""

from typing import List, Dict, Any


def create_waterfall_chart(span_data: List[Dict[str, Any]]):
    """Create Gantt-style waterfall chart from span data.
    
    Only works in Jupyter notebooks. Displays interactive Plotly chart showing:
    - Execution timeline
    - Parallel execution
    - Nested pipelines (hierarchical)
    - Cache hits (green bars)
    
    Args:
        span_data: List of span dictionaries with keys:
            - name: Span name
            - start_time: Start timestamp (seconds)
            - duration: Duration (seconds)
            - depth: Nesting depth
            - cached: Whether cached (bool)
            - type: Type of span ('node', 'pipeline', 'map')
    
    Returns:
        Plotly Figure (auto-displays in Jupyter)
    
    Raises:
        ImportError: If plotly is not installed
    
    Example:
        >>> from hypernodes.telemetry import TelemetryCallback
        >>> 
        >>> telemetry = TelemetryCallback()
        >>> pipeline = Pipeline(nodes=[...], callbacks=[telemetry])
        >>> result = pipeline.run(inputs={...})
        >>> 
        >>> # In Jupyter:
        >>> chart = telemetry.get_waterfall_chart()
        >>> chart  # Auto-displays
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is not installed. Install with: pip install 'hypernodes[telemetry]'"
        )
    
    if not span_data:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No span data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            title="Pipeline Execution Waterfall (No Data)",
            height=300
        )
        return fig
    
    # Normalize start times to 0
    min_time = min(s['start_time'] for s in span_data)
    
    # Sort spans: pipelines first (by depth), then their child nodes
    # This ensures parent pipelines appear above their children in the chart
    sorted_spans = sorted(
        span_data,
        key=lambda s: (
            s.get('depth', 0),  # First by depth (lower depth = higher in hierarchy)
            0 if s.get('type') == 'pipeline' else 1,  # Pipelines before nodes at same depth
            s.get('start_time', 0)  # Then by start time
        )
    )
    
    fig = go.Figure()
    
    # Color mapping
    color_map = {
        'node': 'lightblue',
        'pipeline': 'lightsalmon',
        'map': 'lightgreen'
    }
    
    # Create bars for each span
    for i, span in enumerate(sorted_spans):
        # Calculate positions
        y_pos = len(span_data) - i  # Reverse order for top-to-bottom
        start = (span['start_time'] - min_time) * 1000  # Convert to ms
        duration = span['duration'] * 1000  # Convert to ms
        
        # Color: green if cached, otherwise based on type
        if span.get('cached', False):
            color = 'lightgreen'
            label = f"{span['name']} ⚡ (cached)"
        else:
            color = color_map.get(span.get('type', 'node'), 'lightblue')
            label = span['name']
        
        # Add bar
        fig.add_trace(go.Bar(
            name=label,
            y=[y_pos],
            x=[duration],
            base=start,
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color='gray', width=1)
            ),
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Start: {start:.1f}ms<br>"
                f"Duration: {duration:.2f}ms<br>"
                f"Depth: {span.get('depth', 0)}<br>"
                f"Type: {span.get('type', 'node')}<br>"
                "<extra></extra>"
            ),
            showlegend=False
        ))
        
        # Add text label on the left
        fig.add_annotation(
            x=start - 5,
            y=y_pos,
            text=span['name'],
            showarrow=False,
            xanchor='right',
            font=dict(size=10)
        )
    
    # Update layout
    fig.update_layout(
        title="Pipeline Execution Waterfall",
        xaxis_title="Time (ms)",
        yaxis_title="",
        barmode='overlay',
        height=max(400, len(span_data) * 40),  # Dynamic height
        yaxis=dict(
            showticklabels=False,  # Hide y-axis labels (we use annotations)
            range=[0.5, len(span_data) + 0.5]
        ),
        xaxis=dict(
            range=[
                0,
                max((s['start_time'] - min_time + s['duration']) * 1000 for s in span_data) * 1.1
            ]
        ),
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(l=200, r=50, t=80, b=50)  # Left margin for labels
    )
    
    # Add legend manually (since showlegend=False)
    legend_items = [
        ('Node', 'lightblue'),
        ('Pipeline', 'lightsalmon'),
        ('Map Operation', 'lightgreen'),
        ('Cached', 'lightgreen')
    ]
    
    legend_y = 1.15
    for i, (name, color) in enumerate(legend_items):
        fig.add_annotation(
            x=0.02 + i * 0.25,
            y=legend_y,
            xref='paper',
            yref='paper',
            text=f'<span style="color:{color}">■</span> {name}',
            showarrow=False,
            xanchor='left',
            font=dict(size=10)
        )
    
    return fig
