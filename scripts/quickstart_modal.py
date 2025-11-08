#!/usr/bin/env python3
"""
Quick-start Modal test - run this first!

This is the absolute minimal test to verify Modal is working.
Should complete in ~30-60 seconds (including cold start).
"""

from hypernodes import Pipeline, node
from hypernodes.backend import ModalBackend
import modal


def main():
    print("\n" + "="*60)
    print("MODAL BACKEND QUICK-START TEST")
    print("="*60)
    print("\nThis will:")
    print("1. Create a minimal Modal image")
    print("2. Run a simple pipeline remotely")
    print("3. Verify results are returned correctly")
    print("\nExpected time: ~30-60 seconds (first run)")
    print("="*60 + "\n")
    
    # Simple computation node
    @node(output_name="result")
    def add_one(x: int) -> int:
        """Add 1 to input."""
        return x + 1
    
    # Create minimal Modal image
    print("Creating Modal image...")
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "cloudpickle>=3.0.0"
    )
    print("✓ Image created")
    
    # Create Modal backend
    print("\nCreating Modal backend...")
    backend = ModalBackend(
        image=image,
        timeout=60,
    )
    print("✓ Backend created")
    
    # Create pipeline
    print("\nCreating pipeline...")
    pipeline = Pipeline(nodes=[add_one]).with_engine(backend)
    print("✓ Pipeline created")
    
    # Run on Modal
    print("\nSubmitting to Modal...")
    print("(First run will build image and start container)")
    
    result = pipeline.run(inputs={"x": 41})
    
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"Input:  x = 41")
    print(f"Output: result = {result['result']}")
    
    # Verify
    assert result == {"result": 42}, f"Expected {{'result': 42}}, got {result}"
    
    print("\n" + "="*60)
    print("✓ SUCCESS!")
    print("="*60)
    print("\nModal backend is working correctly!")
    print("\nNext steps:")
    print("1. Run: uv run python scripts/test_modal_smoke.py")
    print("2. Run: uv run pytest tests/test_modal_backend.py -v")
    print("3. Try: uv run python scripts/test_modal_hebrew_minimal.py")
    print("\nSee tests/README_MODAL.md for more information.")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "="*60)
        print("✗ ERROR")
        print("="*60)
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Modal is installed: uv pip install modal")
        print("2. Authenticate: modal token new")
        print("3. Check Modal dashboard: https://modal.com")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        import sys
        sys.exit(1)
