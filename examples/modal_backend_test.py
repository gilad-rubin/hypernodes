"""Simple test of Modal backend with minimal compute.

This example demonstrates:
- Basic Modal backend setup
- Simple pipeline execution on Modal
- Minimal resource usage (no GPU, small CPU)
"""

import modal
from hypernodes import Pipeline, ModalBackend, node


# Create simple test nodes - very lightweight operations
@node(output_name="doubled")
def double(x: int) -> int:
    """Double the input."""
    return x * 2


@node(output_name="result")
def add_ten(doubled: int) -> int:
    """Add ten to the input."""
    return doubled + 10


def main():
    print("=" * 60)
    print("Testing Modal Backend - Minimal Compute")
    print("=" * 60)
    
    # Build a minimal Modal image with required dependencies
    # Copy hypernodes source into the image
    import pathlib
    repo_root = pathlib.Path(__file__).parent.parent
    src_dir = repo_root / "src"
    
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "cloudpickle",
            "networkx",
            "graphviz",  # Required by visualization module
            "rich",
            "tqdm",
        )
        .env({"PYTHONPATH": "/root"})  # Add /root to Python path
        .add_local_dir(str(src_dir), remote_path="/root")  # Copy src/ to /root/ (last!)
    )
    
    # Configure backend with minimal resources (no GPU, default CPU/memory)
    backend = ModalBackend(
        image=image,
        timeout=60,  # Short timeout for simple operations
    )
    
    # Create pipeline
    pipeline = Pipeline(
        nodes=[double, add_ten],
        backend=backend
    )
    
    print("\n1. Testing single run...")
    result = pipeline.run(inputs={"x": 5})
    print(f"   Input: x=5")
    print(f"   Result: {result}")
    assert result["doubled"] == 10, "Double check failed"
    assert result["result"] == 20, "Add ten check failed"
    print("   ✓ Single run succeeded!")
    
    print("\n2. Testing map with 3 items (small batch)...")
    # Use the correct Pipeline.map() API
    results = pipeline.map(
        inputs={"x": [1, 2, 3]},  # x is a list to map over
        map_over="x"  # Map over the x parameter
    )
    print(f"   Results: {results}")
    
    # Verify results - map returns dict of lists
    assert results["doubled"] == [2, 4, 6], "Doubled results incorrect"
    assert results["result"] == [12, 14, 16], "Final results incorrect"
    print("   ✓ Map succeeded!")
    
    print("\n" + "=" * 60)
    print("All tests passed! Modal backend working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
