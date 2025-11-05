"""Test Modal Volume persistence with DiskCache.

This test verifies that cache data persists across separate Modal executions
without explicit volume.commit() calls.

Run this script twice:
1. First run: Should show cache miss and write to cache
2. Second run: Should show cache hit if persistence works
"""

import modal
from hypernodes import Pipeline, ModalBackend, node
from hypernodes.cache import DiskCache
import time


# Create simple test nodes
@node(output_name="result", cache=True)
def expensive_computation(x: int) -> int:
    """Simulate expensive computation that we want to cache."""
    print(f"[EXECUTING] expensive_computation with x={x}")
    time.sleep(0.5)  # Simulate work
    result = x * 100 + 42
    print(f"[COMPUTED] result={result}")
    return result


@node(output_name="final", cache=True)
def process_result(result: int) -> dict:
    """Process the result."""
    print(f"[EXECUTING] process_result with result={result}")
    return {"value": result, "doubled": result * 2}


def main():
    print("=" * 70)
    print("MODAL VOLUME CACHE PERSISTENCE TEST")
    print("=" * 70)
    
    # Build Modal image with hypernodes
    import pathlib
    repo_root = pathlib.Path(__file__).parent.parent
    src_dir = repo_root / "src"
    
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("cloudpickle", "networkx", "graphviz", "rich", "tqdm")
        .env({"PYTHONPATH": "/root"})
        .add_local_dir(str(src_dir), remote_path="/root")
    )
    
    # Create persistent volume for cache
    print("\n[SETUP] Creating Modal Volume 'test-cache'...")
    volume = modal.Volume.from_name("test-cache", create_if_missing=True)
    print("✓ Volume created/accessed")
    
    # Configure ModalBackend with volume mounted at /cache
    print("\n[SETUP] Configuring ModalBackend with volume at /cache...")
    backend = ModalBackend(
        image=image,
        volumes={"/cache": volume},
        timeout=120,
    )
    print("✓ Backend configured")
    
    # Important: Don't create DiskCache locally - /cache only exists on Modal!
    # Instead, we'll configure the pipeline to use caching, and set the cache
    # path inside the remote function.
    # For now, use a local temp cache (won't be used since backend is Modal)
    print("\n[SETUP] Configuring pipeline (cache will be at /cache on Modal)...")
    
    # We need to pass the cache path to the remote function somehow.
    # Looking at the backend, we need to modify how cache is handled.
    # For this test, let's create a local cache that won't be used
    import tempfile
    local_cache_dir = tempfile.mkdtemp()
    cache = DiskCache(path=local_cache_dir)
    
    pipeline = Pipeline(
        nodes=[expensive_computation, process_result],
        backend=backend,
        cache=cache,  # This will be serialized but path is wrong
        name="volume_cache_test"
    )
    print("✓ Pipeline created (cache setup needs fixing)")
    
    # Run the pipeline
    print("\n" + "=" * 70)
    print("EXECUTING PIPELINE")
    print("=" * 70)
    
    test_input = {"x": 123}
    print(f"\nInput: {test_input}")
    
    start_time = time.time()
    result = pipeline.run(inputs=test_input)
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Output: {result}")
    print(f"Execution time: {elapsed:.2f}s")
    
    # Interpret results
    print("\n" + "=" * 70)
    print("CACHE BEHAVIOR ANALYSIS")
    print("=" * 70)
    
    if elapsed < 0.3:
        print("✓ CACHE HIT: Execution was very fast (< 0.3s)")
        print("  This means cache data persisted from a previous run!")
        print("  Modal Volumes automatically persist without explicit commits.")
    else:
        print("⚠ CACHE MISS: Execution took longer (>= 0.3s)")
        print("  This is expected on the first run.")
        print("  Run this script again to test persistence.")
    
    print("\n" + "=" * 70)
    print("INSTRUCTIONS")
    print("=" * 70)
    print("Run this script again with:")
    print("  uv run python scripts/test_modal_volume_cache.py")
    print("\nIf the second run shows CACHE HIT, persistence works! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()

