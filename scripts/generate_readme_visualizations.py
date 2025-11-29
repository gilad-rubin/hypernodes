"""
Generate visualizations for the README to demonstrate:
1. Think Singular - Simple pipeline for one item (named "text_processor")
2. Compose - Nest that same pipeline in a larger workflow
3. Scale - .map() over many items
"""

import re
from pathlib import Path

from hypernodes import Pipeline, node

# ============================================================================
# STEP 1: Think Singular - A simple text processing pipeline
# ============================================================================

@node(output_name="cleaned")
def clean(text: str) -> str:
    """Clean a single piece of text."""
    return text.strip().lower()


@node(output_name="word_count")
def count(cleaned: str) -> int:
    """Count words."""
    return len(cleaned.split())


# Give it a clear name that will be visible when nested!
text_processor = Pipeline(nodes=[clean, count], name="text_processor")


# ============================================================================
# STEP 2: Compose - Use text_processor as a node in a larger pipeline
# ============================================================================

@node(output_name="report")
def summarize(word_count: int, cleaned: str) -> str:
    """Generate a summary report."""
    return f"Processed: {word_count} words"


# Compose: text_processor becomes a node in the analysis pipeline
analysis_pipeline = Pipeline(
    nodes=[text_processor.as_node(), summarize],
    name="analysis"
)


# ============================================================================
# STEP 3: Scale - Process many items with map_over
# ============================================================================

# Use text_processor to process batches
batch_processor = text_processor.as_node(
    name="batch_text_processor",
    map_over="texts",
    input_mapping={"texts": "text"},
    output_mapping={"word_count": "word_counts"}
)


@node(output_name="total")
def aggregate(word_counts: list) -> int:
    """Sum all word counts."""
    return sum(word_counts)


batch_pipeline = Pipeline(
    nodes=[batch_processor, aggregate],
    name="batch_analysis"
)


def extract_svg_content(html_obj):
    """Extract SVG from IPython HTML object."""
    html_str = html_obj.data
    # Find the SVG content
    svg_match = re.search(r'(<svg.*?</svg>)', html_str, re.DOTALL)
    if svg_match:
        return svg_match.group(1)
    return html_str


def save_visualization(pipeline, filename, **kwargs):
    """Generate and save a visualization."""
    html_obj = pipeline.visualize(**kwargs)
    svg_content = extract_svg_content(html_obj)
    
    out_dir = Path(__file__).parent.parent / "assets" / "readme"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = out_dir / filename
    with open(filepath, "w") as f:
        f.write(svg_content)
    print(f"Saved: {filepath}")
    return filepath


def main():
    print("Generating README visualizations...\n")
    
    # 1. Simple single-item pipeline (Think Singular) - named "text_processor"
    print("1. text_processor pipeline (single item):")
    save_visualization(
        text_processor,
        "step1_think_singular.svg",
        depth=1
    )
    
    # 2. Composed pipeline collapsed - shows text_processor as a node
    print("\n2. analysis pipeline (collapsed - text_processor as node):")
    save_visualization(
        analysis_pipeline,
        "step2_compose_collapsed.svg",
        depth=1
    )
    
    # 3. Composed pipeline expanded - shows text_processor internals
    print("\n3. analysis pipeline (expanded - showing text_processor internals):")
    save_visualization(
        analysis_pipeline,
        "step3_compose_expanded.svg",
        depth=2
    )
    
    # 4. Batch pipeline - shows map_over scaling
    print("\n4. batch_analysis pipeline (scaling with map_over):")
    save_visualization(
        batch_pipeline,
        "step4_scale.svg",
        depth=1
    )
    
    print("\n✅ All visualizations generated!")
    print(f"\nOutput directory: {Path(__file__).parent.parent / 'assets' / 'readme'}")


if __name__ == "__main__":
    main()
