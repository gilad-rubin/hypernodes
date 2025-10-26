"""Generate sample visualizations showcasing different styles."""
import sys
sys.path.insert(0, 'src')

from hypernodes import node, Pipeline, DESIGN_STYLES
from typing import List, Dict, Any


def create_sample_pipeline():
    """Create a sample text processing pipeline."""
    @node(output_name="cleaned")
    def clean_text(text: str, lowercase: bool = True) -> str:
        """Clean and normalize text."""
        result = text.strip()
        if lowercase:
            result = result.lower()
        return result

    @node(output_name="tokens")
    def tokenize(cleaned: str) -> List[str]:
        """Split text into tokens."""
        return cleaned.split()

    @node(output_name="word_count")
    def count_words(tokens: List[str]) -> int:
        """Count number of words."""
        return len(tokens)

    @node(output_name="stats")
    def compute_stats(tokens: List[str], word_count: int) -> Dict[str, Any]:
        """Compute text statistics."""
        avg_length = sum(len(token) for token in tokens) / max(word_count, 1)
        return {"word_count": word_count, "avg_word_length": avg_length}

    return Pipeline(nodes=[clean_text, tokenize, count_words, compute_stats], name="text_analysis")


def create_hierarchical_pipeline():
    """Create a hierarchical pipeline with nested sub-pipelines."""
    # Inner preprocessing pipeline
    @node(output_name="lowercased")
    def to_lowercase(text: str) -> str:
        return text.lower()

    @node(output_name="trimmed")
    def trim_whitespace(lowercased: str) -> str:
        return lowercased.strip()

    preprocess = Pipeline(nodes=[to_lowercase, trim_whitespace], name="preprocess")

    # Analysis pipeline
    @node(output_name="split_tokens")
    def split(trimmed: str) -> List[str]:
        return trimmed.split()

    @node(output_name="token_count")
    def count(split_tokens: List[str]) -> int:
        return len(split_tokens)

    analysis = Pipeline(nodes=[split, count], name="analyze")

    # Final aggregation
    @node(output_name="summary")
    def summarize(trimmed: str, token_count: int) -> Dict[str, Any]:
        return {"text": trimmed, "count": token_count}

    return Pipeline(nodes=[preprocess, analysis, summarize], name="hierarchical")


def main():
    """Generate sample visualizations."""
    print("Generating sample visualizations...\n")
    
    simple = create_sample_pipeline()
    hierarchical = create_hierarchical_pipeline()
    
    # Generate simple pipeline with different styles
    print("1. Simple Pipeline - Different Styles:")
    for style_name in ["default", "minimal", "professional", "vibrant", "pastel"]:
        filename = f"outputs/simple_{style_name}.svg"
        simple.visualize(filename=filename, style=style_name, show_legend=True)
        print(f"   ✓ Generated {filename}")
    
    # Generate with different orientations
    print("\n2. Simple Pipeline - Different Orientations:")
    for orient in ["TB", "LR"]:
        filename = f"outputs/simple_orient_{orient}.svg"
        simple.visualize(filename=filename, style="professional", orient=orient)
        print(f"   ✓ Generated {filename}")
    
    # Generate hierarchical with different depths
    print("\n3. Hierarchical Pipeline - Different Depths:")
    for depth in [1, 2, None]:
        depth_label = depth if depth is not None else "full"
        filename = f"outputs/hierarchical_depth_{depth_label}.svg"
        hierarchical.visualize(filename=filename, style="professional", depth=depth, show_legend=True)
        print(f"   ✓ Generated {filename}")
    
    # Generate hierarchical with different styles
    print("\n4. Hierarchical Pipeline - Different Styles (expanded):")
    for style_name in ["default", "vibrant", "dark", "monochrome"]:
        filename = f"outputs/hierarchical_{style_name}.svg"
        hierarchical.visualize(filename=filename, style=style_name, depth=None, orient="LR")
        print(f"   ✓ Generated {filename}")
    
    # Generate with/without type hints
    print("\n5. Type Hints Comparison:")
    for show_types in [True, False]:
        types_label = "with" if show_types else "without"
        filename = f"outputs/simple_{types_label}_types.svg"
        simple.visualize(filename=filename, style="professional", show_types=show_types)
        print(f"   ✓ Generated {filename}")
    
    # Generate with/without grouping
    print("\n6. Input Grouping Comparison:")
    for group_size in [None, 2]:
        group_label = "ungrouped" if group_size is None else "grouped"
        filename = f"outputs/simple_{group_label}.svg"
        simple.visualize(filename=filename, style="professional", min_arg_group_size=group_size)
        print(f"   ✓ Generated {filename}")
    
    print("\n" + "="*60)
    print(f"All visualizations saved to outputs/ directory!")
    print(f"Total files: {5 + 2 + 3 + 4 + 2 + 2} SVG files")
    print("="*60)
    print("\nYou can:")
    print("  1. Open them in your browser to view")
    print("  2. Compare different styles and configurations")
    print("  3. Choose your favorite design!")


if __name__ == "__main__":
    main()
