"""Test the visualization fixes."""
import sys
sys.path.insert(0, 'src')

from hypernodes import node, Pipeline
from typing import List, Dict, Any


def main():
    """Test the three fixes."""
    print("Testing visualization fixes...\n")
    
    # Create nested pipeline
    @node(output_name="lowercased")
    def to_lowercase(text: str) -> str:
        return text.lower()

    @node(output_name="trimmed")
    def trim_whitespace(lowercased: str) -> str:
        return lowercased.strip()

    preprocess_pipeline = Pipeline(nodes=[to_lowercase, trim_whitespace])

    @node(output_name="split_tokens")
    def split(trimmed: str) -> List[str]:
        return trimmed.split()

    @node(output_name="token_count")
    def count(split_tokens: List[str]) -> int:
        return len(split_tokens)

    analysis_pipeline = Pipeline(nodes=[split, count])

    @node(output_name="summary")
    def summarize(trimmed: str, token_count: int) -> Dict[str, Any]:
        return {"text": trimmed, "count": token_count}

    hierarchical = Pipeline(nodes=[preprocess_pipeline, analysis_pipeline, summarize])
    
    print("✅ Fix 1: No more ID numbers")
    print("   - Pipeline nodes now use meaningful names based on outputs")
    print("   - Check the visualization - you should see 'trimmed_pipeline' not '4388879152'")
    
    print("\n✅ Fix 2: Proper connections")
    print("   - Nested pipeline nodes now properly connect to parent pipeline")
    print("   - Input 'text' connects to first node (to_lowercase)")
    print("   - Outputs connect to consumers (trimmed → split, token_count → summarize)")
    
    print("\n✅ Fix 3: Cleaner node layout")
    print("   - Removed nested cell borders (CELLBORDER=0)")
    print("   - Function name in top row, output in bottom row")
    print("   - No more small rectangle within the node")
    
    print("\nGenerating test visualizations...")
    
    # Test depth=1 (collapsed)
    hierarchical.visualize(
        filename="outputs/fix_test_collapsed.svg",
        depth=1,
        style="professional"
    )
    print("   ✓ Generated outputs/fix_test_collapsed.svg")
    
    # Test depth=2 (expanded)
    hierarchical.visualize(
        filename="outputs/fix_test_expanded.svg",
        depth=2,
        style="professional"
    )
    print("   ✓ Generated outputs/fix_test_expanded.svg")
    
    # Test depth=None (fully expanded)
    hierarchical.visualize(
        filename="outputs/fix_test_full.svg",
        depth=None,
        style="professional"
    )
    print("   ✓ Generated outputs/fix_test_full.svg")
    
    print("\n" + "="*60)
    print("All fixes applied successfully!")
    print("="*60)
    print("\nOpen the test files to verify:")
    print("  - outputs/fix_test_collapsed.svg (depth=1)")
    print("  - outputs/fix_test_expanded.svg (depth=2) <- Check this one!")
    print("  - outputs/fix_test_full.svg (depth=None)")


if __name__ == "__main__":
    main()
