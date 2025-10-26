"""Test Modal backend map operations."""
import pytest

pytest.importorskip("modal")

import modal
from pathlib import Path
import os

from hypernodes import ModalBackend, Pipeline, node
from hypernodes.telemetry import ProgressCallback


@node(output_name="doubled")
def double(x: int) -> int:
    """Double the input."""
    return x * 2


@node(output_name="result")
def add_ten(doubled: int) -> int:
    """Add ten to the input."""
    return doubled + 10


def test_modal_map_operation():
    """Test that map operations work with Modal backend."""
    # Get repo root and src directory
    repo_root = Path(os.getcwd())
    src_dir = repo_root / "src"
    
    # Create Modal image
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "cloudpickle",
            "networkx",
            "graphviz",
            "rich",
            "tqdm",
            "ipywidgets",
        )
        .env({"PYTHONPATH": "/root"})
        .add_local_dir(str(src_dir), remote_path="/root")
    )
    
    # Configure backend
    backend = ModalBackend(
        image=image,
        timeout=60,
    )
    
    # Create pipeline with callbacks
    pipeline = Pipeline(
        nodes=[double, add_ten],
        callbacks=[ProgressCallback()],
        backend=backend,
    )
    
    # Test single run first
    result = pipeline.run(inputs={"x": 5})
    assert result["doubled"] == 10
    assert result["result"] == 20
    
    # Test map operation - this should fail before the fix
    results = pipeline.map(
        inputs={"x": [1, 2, 3]},
        map_over="x",
    )
    
    # Verify results
    assert results["doubled"] == [2, 4, 6]
    assert results["result"] == [12, 14, 16]


if __name__ == "__main__":
    test_modal_map_operation()
