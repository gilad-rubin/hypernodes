#!/usr/bin/env python3
"""
Diagnose Modal connection issues.

This script helps identify what's causing the connection timeouts.
"""

import modal
from hypernodes import Pipeline, node
from hypernodes.backend import ModalBackend
import time


def test_1_basic_import():
    """Test if hypernodes imports work on Modal."""
    print("\n" + "="*60)
    print("TEST 1: Basic Import Test")
    print("="*60)
    
    # Minimal image with just hypernodes
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "cloudpickle>=3.0.0",
    ).run_commands(
        "pip install -e /root/hypernodes || echo 'Install failed'"
    ).add_local_dir(
        "/Users/giladrubin/python_workspace/hypernodes", 
        remote_path="/root/hypernodes"
    )
    
    app = modal.App("test-imports")
    
    @app.function(image=image, timeout=60)
    def test_import():
        try:
            from hypernodes import Pipeline, node
            from hypernodes.backend import LocalBackend
            return {"status": "success", "message": "Imports work!"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    with app.run():
        result = test_import.remote()
        print(f"Result: {result}")
        
        if result["status"] == "success":
            print("✓ PASSED - Imports work on Modal")
            return True
        else:
            print(f"✗ FAILED - {result['message']}")
            return False


def test_2_simple_pipeline():
    """Test simplest pipeline on Modal."""
    print("\n" + "="*60)
    print("TEST 2: Simple Pipeline")
    print("="*60)
    
    @node(output_name="result")
    def add_one(x: int) -> int:
        return x + 1
    
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "cloudpickle>=3.0.0",
    )
    
    backend = ModalBackend(image=image, timeout=60)
    pipeline = Pipeline(nodes=[add_one]).with_backend(backend)
    
    try:
        result = pipeline.run(inputs={"x": 5})
        print(f"Result: {result}")
        assert result == {"result": 6}
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_slow_execution():
    """Test if timeout is the issue."""
    print("\n" + "="*60)
    print("TEST 3: Slow Execution (30 seconds)")
    print("="*60)
    
    @node(output_name="result")
    def slow_task(x: int) -> int:
        import time
        print("Starting slow task...")
        for i in range(6):
            print(f"Progress: {i*5} seconds")
            time.sleep(5)
        print("Task complete!")
        return x * 2
    
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "cloudpickle>=3.0.0",
    )
    
    backend = ModalBackend(image=image, timeout=120)  # 2 minute timeout
    pipeline = Pipeline(nodes=[slow_task]).with_backend(backend)
    
    try:
        print("Running 30-second task on Modal...")
        start = time.time()
        result = pipeline.run(inputs={"x": 21})
        duration = time.time() - start
        
        print(f"Result: {result}")
        print(f"Duration: {duration:.1f}s")
        assert result == {"result": 42}
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_image_with_dependencies():
    """Test if the full image works."""
    print("\n" + "="*60)
    print("TEST 4: Full Image with Dependencies")
    print("="*60)
    
    @node(output_name="result")
    def test_imports(x: int) -> dict:
        """Test that all packages are available."""
        imports_ok = []
        imports_failed = []
        
        packages = [
            "cloudpickle",
            "numpy",
            "pandas",
            "pydantic",
        ]
        
        for pkg in packages:
            try:
                __import__(pkg)
                imports_ok.append(pkg)
            except ImportError:
                imports_failed.append(pkg)
        
        return {
            "x_doubled": x * 2,
            "imports_ok": imports_ok,
            "imports_failed": imports_failed,
        }
    
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "cloudpickle>=3.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pydantic>=2.0.0",
    )
    
    backend = ModalBackend(image=image, timeout=120)
    pipeline = Pipeline(nodes=[test_imports]).with_backend(backend)
    
    try:
        result = pipeline.run(inputs={"x": 21})
        print(f"Result: {result}")
        print(f"  Imports OK: {', '.join(result['imports_ok'])}")
        if result['imports_failed']:
            print(f"  Imports FAILED: {', '.join(result['imports_failed'])}")
        
        assert result["x_doubled"] == 42
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_with_progress():
    """Test with progress reporting."""
    print("\n" + "="*60)
    print("TEST 5: With Progress Reporting")
    print("="*60)
    
    @node(output_name="results")
    def process_items(items: list) -> list:
        """Process items with progress."""
        results = []
        for i, item in enumerate(items):
            print(f"Processing item {i+1}/{len(items)}")
            results.append(item * 2)
        return results
    
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "cloudpickle>=3.0.0",
    )
    
    backend = ModalBackend(
        image=image,
        timeout=120,
    )
    pipeline = Pipeline(nodes=[process_items]).with_backend(backend)
    
    try:
        result = pipeline.run(inputs={"items": [1, 2, 3, 4, 5]})
        print(f"Result: {result}")
        assert result == {"results": [2, 4, 6, 8, 10]}
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run diagnostic tests."""
    print("\n" + "="*60)
    print("MODAL BACKEND DIAGNOSTIC TESTS")
    print("="*60)
    print("\nThese tests help diagnose connection issues.")
    print("Run them in order to identify the problem.")
    
    tests = [
        # test_1_basic_import,  # Skip - complex setup
        test_2_simple_pipeline,
        test_3_slow_execution,
        test_4_image_with_dependencies,
        test_5_with_progress,
    ]
    
    results = []
    for test in tests:
        try:
            passed = test()
            results.append((test.__name__, passed))
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    print(f"\nPassed: {passed_count}/{len(results)}")
    
    if passed_count < len(results):
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        # Check which test failed
        for name, passed in results:
            if not passed:
                if "simple_pipeline" in name:
                    print("\n✗ Basic pipeline failed")
                    print("  - Check Modal authentication: modal token new")
                    print("  - Check Modal dashboard for errors")
                    print("  - Try: modal app list")
                
                elif "slow_execution" in name:
                    print("\n✗ Slow execution failed")
                    print("  - This suggests timeout issues")
                    print("  - Your Hebrew pipeline might be too slow for current timeout")
                    print("  - Try increasing timeout parameter")
                    print("  - Consider breaking pipeline into smaller chunks")
                
                elif "dependencies" in name:
                    print("\n✗ Dependencies failed")
                    print("  - Image build might be incomplete")
                    print("  - Check which imports failed")
                    print("  - Verify package versions in image definition")
                
                elif "progress" in name:
                    print("\n✗ Progress reporting failed")
                    print("  - Connection might be dropping during execution")
                    print("  - Check Modal logs for more details")
