#!/usr/bin/env python3
"""
Fixed Modal setup for Hebrew retrieval pipeline.

Key fixes:
1. Properly install hypernodes in the Modal image
2. Add sys.path modification to ensure imports work
3. Set appropriate timeouts
"""

import os
from pathlib import Path
import modal

# Paths
repo_root = Path(os.getcwd())
src_dir = repo_root / "src"
hypernodes_dir = Path("/Users/giladrubin/python_workspace/hypernodes")

# Create Modal volumes
models_volume = modal.Volume.from_name("mafat-models", create_if_missing=True)
data_volume = modal.Volume.from_name("mafat-data", create_if_missing=True)

# Define Modal image with proper hypernodes installation
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env(
        {
            "HF_HOME": "/root/models",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # Install core dependencies first
    .uv_pip_install(
        # Core dependencies
        "cloudpickle>=3.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pydantic>=2.0.0",
        # Retrieval & ML
        "pylate",
        "transformers",
        "sentence-transformers",
        "FlagEmbedding",
        "torch",
        "optimum",
        "rank-bm25",
        # Data & Evaluation
        "pyarrow",
        "pytrec_eval",
        # Utilities
        "diskcache",
        "networkx",
        "graphviz",
        "rich",
        "tqdm",
    )
    # Add hypernodes source code
    .add_local_dir(
        str(hypernodes_dir / "src" / "hypernodes"),
        remote_path="/root/hypernodes_pkg/hypernodes",
    )
    # Install hypernodes as package (create a minimal setup.py on the fly)
    .run_commands(
        "cd /root/hypernodes_pkg && "
        "echo 'from setuptools import setup, find_packages; "
        "setup(name=\"hypernodes\", packages=find_packages())' > setup.py && "
        "pip install -e ."
    )
)


# Alternative: Simpler approach using sys.path
image_simple = (
    modal.Image.debian_slim(python_version="3.12")
    .env(
        {
            "HF_HOME": "/root/models",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/hypernodes/src:$PYTHONPATH",  # Add to path
        }
    )
    .uv_pip_install(
        "cloudpickle>=3.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pydantic>=2.0.0",
        "pylate",
        "transformers",
        "sentence-transformers",
        "FlagEmbedding",
        "torch",
        "optimum",
        "rank-bm25",
        "pyarrow",
        "pytrec_eval",
        "diskcache",
        "networkx",
        "graphviz",
        "rich",
        "tqdm",
    )
    .add_local_dir(
        str(hypernodes_dir / "src"),
        remote_path="/root/hypernodes/src",
    )
)


def create_modal_backend_recommended():
    """Create Modal backend with recommended settings."""
    from hypernodes.backend import ModalBackend
    
    backend = ModalBackend(
        image=image_simple,  # Use the simpler image with PYTHONPATH
        gpu="A10G",
        timeout=7200,  # 2 hours - adjust based on your needs
        map_execution="sequential",  # Start with sequential
        volumes={
            "/root/models": models_volume,
            "/root/data": data_volume,
        },
    )
    return backend


def test_modal_setup():
    """Test that Modal setup works."""
    from hypernodes import Pipeline, node
    from hypernodes.backend import ModalBackend
    
    @node(output_name="result")
    def test_imports() -> dict:
        """Test that hypernodes imports work."""
        try:
            from hypernodes import Pipeline, node
            from hypernodes.backend import LocalBackend, PipelineExecutionEngine
            from hypernodes.callbacks import CallbackContext
            
            return {
                "status": "success",
                "message": "All imports work!",
                "hypernodes_available": True,
            }
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
                "hypernodes_available": False,
            }
    
    backend = ModalBackend(
        image=image_simple,
        timeout=120,
    )
    
    pipeline = Pipeline(nodes=[test_imports]).with_backend(backend)
    
    print("Testing Modal setup...")
    result = pipeline.run(inputs={})
    
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    
    if result['status'] == 'error':
        print("\nTraceback:")
        print(result['traceback'])
        print("\n✗ FAILED - Hypernodes imports don't work on Modal")
        return False
    else:
        print("\n✓ SUCCESS - Modal setup is working correctly!")
        return True


if __name__ == "__main__":
    """Test the Modal setup."""
    import sys
    
    print("\n" + "="*60)
    print("MODAL SETUP TEST FOR HEBREW PIPELINE")
    print("="*60)
    print("\nThis will test if hypernodes works on Modal.")
    print("If successful, you can use this image for your pipeline.")
    
    try:
        success = test_modal_setup()
        
        if success:
            print("\n" + "="*60)
            print("NEXT STEPS")
            print("="*60)
            print("\n1. Update your pipeline code to use:")
            print("   from scripts.fix_modal_hebrew_example import create_modal_backend_recommended")
            print("   backend = create_modal_backend_recommended()")
            print("   pipeline = pipeline.with_backend(backend)")
            print("\n2. Run your Hebrew retrieval pipeline")
            print("\n3. If it still times out, increase timeout parameter")
            sys.exit(0)
        else:
            print("\n" + "="*60)
            print("TROUBLESHOOTING")
            print("="*60)
            print("\n1. Check Modal authentication: modal token new")
            print("2. Verify paths are correct")
            print("3. Check Modal dashboard for error logs")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
