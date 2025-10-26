"""Test script to verify Modal backend map operation fix."""
import modal
import os
from pathlib import Path

from hypernodes import ModalBackend, Pipeline, node
from hypernodes.telemetry import ProgressCallback


# Create simple test nodes - very lightweight operations
@node(output_name="doubled")
def double(x: int) -> int:
    """Double the input."""
    return x * 2


@node(output_name="result")
def add_ten(doubled: int) -> int:
    """Add ten to the input."""
    return doubled + 10


print("=" * 60)
print("Testing Modal Backend - Map Operation Fix")
print("=" * 60)

# Create pipeline with callbacks
pipeline = Pipeline(
    nodes=[double, add_ten], 
    callbacks=[ProgressCallback()]
)

# Setup Modal image
repo_root = Path(os.getcwd())
src_dir = repo_root / "src"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "cloudpickle",
        "networkx",
        "graphviz",  # Required by visualization module
        "rich",
        "tqdm",
        "ipywidgets",
    )
    .env({"PYTHONPATH": "/root"})  # Add /root to Python path
    .add_local_dir(str(src_dir), remote_path="/root")  # Copy src/ to /root/ (last!)
)

# Configure backend with minimal resources (no GPU, default CPU/memory)
backend = ModalBackend(
    image=image,
    timeout=60,  # Short timeout for simple operations
)

# Attach backend to pipeline
pipeline.with_backend(backend)

print("\n1. Testing single run...")
result = pipeline.run(inputs={"x": 5})
print("   Input: x=5")
print(f"   Result: {result}")
assert result["doubled"] == 10, "Double check failed"
assert result["result"] == 20, "Add ten check failed"
print("   ✓ Single run succeeded!")

print("\n2. Testing map with 3 items (small batch)...")
# Use the correct Pipeline.map() API
results = pipeline.map(
    inputs={"x": [1, 2, 3]},  # x is a list to map over
    map_over="x",  # Map over the x parameter
)
print(f"   Results: {results}")

# Verify results - map returns dict of lists
assert results["doubled"] == [2, 4, 6], "Doubled results incorrect"
assert results["result"] == [12, 14, 16], "Final results incorrect"
print("   ✓ Map succeeded!")

print("\n" + "=" * 60)
print("All tests passed! Modal backend working correctly.")
print("=" * 60)
