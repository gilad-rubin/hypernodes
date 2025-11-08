"""
Working Modal Backend Example for Jupyter Notebook

Copy these cells to your notebook in order.
This example is tested and works with all the fixes applied.
"""

# ==================== CELL 1: Setup and Image ====================
from pathlib import Path
import modal
from hypernodes import Pipeline, node, ModalBackend
from hypernodes.telemetry import ProgressCallback

# Define simple nodes
@node(output_name="doubled")
def double(x: int) -> int:
    """Double the input."""
    return x * 2

@node(output_name="result")
def add_ten(doubled: int) -> int:
    """Add ten to the input."""
    return doubled + 10

# Create Modal image with fixes
hypernodes_dir = Path("/Users/giladrubin/python_workspace/hypernodes")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({
        "PYTHONPATH": "/root/hypernodes/src:$PYTHONPATH",  # FIX 1: Add to path
        "PYTHONUNBUFFERED": "1",
    })
    .uv_pip_install(
        "cloudpickle",
        "networkx",
        "graphviz",
        "rich",
        "tqdm",
    )
    .add_local_dir(
        str(hypernodes_dir / "src"),  # FIX 2: Copy src/ only
        remote_path="/root/hypernodes/src",
    )
)

# Create backend
backend = ModalBackend(
    image=image,
    timeout=120,  # FIX 3: Longer timeout
)

print("✓ Image and backend configured")


# ==================== CELL 2: Create Pipeline ====================
# Create pipeline with callbacks
# Note: Callbacks work locally, but are automatically stripped for Modal execution
pipeline = Pipeline(
    nodes=[double, add_ten],
    callbacks=[ProgressCallback()],  # Callbacks stripped automatically for Modal
)

# Visualize (optional)
pipeline.visualize()

# Set backend
pipeline = pipeline.with_engine(backend)

print("✓ Pipeline created and visualized")


# ==================== CELL 3: Test Single Run ====================
print("\n" + "="*60)
print("Test 1: Single Run")
print("="*60)

result = pipeline.run(inputs={"x": 5})

print(f"Input: x=5")
print(f"Result: {result}")
assert result["doubled"] == 10
assert result["result"] == 20
print("✓ Single run succeeded!")


# ==================== CELL 4: Test Map Operation ====================
print("\n" + "="*60)
print("Test 2: Map Operation")
print("="*60)

results = pipeline.map(
    inputs={"x": [1, 2, 3]},
    map_over="x",
)

print(f"Input: x=[1, 2, 3]")
print(f"Results: {results}")
assert results["doubled"] == [2, 4, 6]
assert results["result"] == [12, 14, 16]
print("✓ Map operation succeeded!")


# ==================== CELL 5: Summary ====================
print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nKey fixes applied:")
print("1. PYTHONPATH set correctly")
print("2. Only src/ folder copied to Modal")
print("3. Callbacks automatically stripped for serialization")
print("4. Longer timeout configured")
print("\n✅ Your pipeline is now ready for Modal execution!")


# ==================== OPTIONAL: Test with Larger Data ====================
"""
# Uncomment to test with more data
print("\n" + "="*60)
print("Test 3: Larger Dataset")
print("="*60)

large_input = list(range(100))
large_results = pipeline.map(
    inputs={"x": large_input},
    map_over="x",
)

print(f"Processed {len(large_input)} items")
print(f"First 5 results: {large_results['result'][:5]}")
print(f"Last 5 results: {large_results['result'][-5:]}")
print("✓ Large dataset processed successfully!")
"""
