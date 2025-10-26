"""Test Modal backend with PyTorch tensors - CUDA to CPU deserialization."""
import pytest

pytest.importorskip("modal")
pytest.importorskip("torch")

import modal
from pathlib import Path
import os
import torch

from hypernodes import ModalBackend, Pipeline, node


@node(output_name="tensor")
def create_tensor(x: int) -> torch.Tensor:
    """Create a tensor (will be on CUDA if GPU available)."""
    tensor = torch.tensor([x, x * 2, x * 3], dtype=torch.float32)
    # If CUDA is available in Modal, tensor will be on GPU
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


@node(output_name="result")
def process_tensor(tensor: torch.Tensor) -> dict:
    """Process tensor and return info."""
    return {
        "sum": float(tensor.sum().item()),
        "device": str(tensor.device),
        "shape": list(tensor.shape),
    }


def test_modal_torch_cpu_deserialization():
    """Test that PyTorch tensors from GPU are correctly mapped to CPU locally."""
    # Get repo root and src directory
    repo_root = Path(os.getcwd())
    src_dir = repo_root / "src"
    
    # Create Modal image with PyTorch
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "cloudpickle",
            "networkx",
            "graphviz",
            "torch",
        )
        .env({"PYTHONPATH": "/root"})
        .add_local_dir(str(src_dir), remote_path="/root")
    )
    
    # Configure backend with GPU (tensors will be on CUDA remotely)
    backend = ModalBackend(
        image=image,
        gpu="any",  # Use any available GPU
        timeout=60,
    )
    
    # Create pipeline
    pipeline = Pipeline(
        nodes=[create_tensor, process_tensor],
        backend=backend,
    )
    
    # Run pipeline - this should work even if local machine has no GPU
    result = pipeline.run(inputs={"x": 5})
    
    # Verify results
    assert "tensor" in result
    assert "result" in result
    assert result["result"]["sum"] == 30.0  # 5 + 10 + 15
    assert result["result"]["shape"] == [3]
    
    # The tensor should now be on CPU locally (even if it was on CUDA remotely)
    tensor = result["tensor"]
    assert isinstance(tensor, torch.Tensor)
    # On local machine without GPU, tensor should be on CPU
    if not torch.cuda.is_available():
        assert tensor.device.type == "cpu"
    
    print(f"✓ Test passed! Tensor device: {tensor.device}")


if __name__ == "__main__":
    test_modal_torch_cpu_deserialization()
