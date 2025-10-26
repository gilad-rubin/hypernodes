"""Test script for pipeline visualization."""
import sys
sys.path.insert(0, 'src')

from hypernodes import node, Pipeline, DESIGN_STYLES
from typing import List, Dict, Any


def main():
    """Test visualization functionality."""
    print("Creating test pipeline...")
    
    # Create a simple pipeline
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

    pipeline = Pipeline(nodes=[clean_text, tokenize, count_words, compute_stats])
    print(f"✓ Pipeline created with {len(pipeline.nodes)} nodes")
    
    # Test available styles
    print(f"\n✓ Available styles: {list(DESIGN_STYLES.keys())}")
    
    # Test basic visualization (returns graphviz object)
    print("\nTesting visualizations...")
    
    for style_name in ["default", "minimal", "professional"]:
        try:
            result = pipeline.visualize(style=style_name, return_type="graphviz")
            print(f"✓ {style_name} style works")
        except Exception as e:
            print(f"✗ {style_name} style failed: {e}")
    
    # Test orientations
    print("\nTesting orientations...")
    for orient in ["TB", "LR"]:
        try:
            result = pipeline.visualize(orient=orient, return_type="graphviz")
            print(f"✓ Orientation {orient} works")
        except Exception as e:
            print(f"✗ Orientation {orient} failed: {e}")
    
    # Test depth levels with nested pipeline
    print("\nTesting nested pipeline...")
    
    @node(output_name="preprocessed")
    def preprocess(text: str) -> str:
        return text.lower().strip()
    
    @node(output_name="processed")
    def process(preprocessed: str) -> str:
        return preprocessed.upper()
    
    inner_pipeline = Pipeline(nodes=[preprocess, process])
    
    @node(output_name="final")
    def finalize(processed: str) -> str:
        return f"Result: {processed}"
    
    outer_pipeline = Pipeline(nodes=[inner_pipeline, finalize])
    
    try:
        # Test collapsed
        result = outer_pipeline.visualize(depth=1, return_type="graphviz")
        print("✓ Nested pipeline (depth=1) works")
        
        # Test expanded
        result = outer_pipeline.visualize(depth=None, return_type="graphviz")
        print("✓ Nested pipeline (depth=None) works")
    except Exception as e:
        print(f"✗ Nested pipeline failed: {e}")
    
    # Test file export
    print("\nTesting file export...")
    try:
        import os
        os.makedirs("outputs", exist_ok=True)
        pipeline.visualize(filename="outputs/test_pipeline.svg", style="professional")
        if os.path.exists("outputs/test_pipeline.svg"):
            print("✓ File export works")
            os.remove("outputs/test_pipeline.svg")
        else:
            print("✗ File export failed: file not created")
    except Exception as e:
        print(f"✗ File export failed: {e}")
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)


if __name__ == "__main__":
    main()
