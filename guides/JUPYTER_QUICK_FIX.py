"""
QUICK FIX for Jupyter + Modal Connection Issues

Copy this entire cell to the TOP of your notebook, BEFORE your pipeline definition.
This replaces your current Modal image setup.
"""

from pathlib import Path
import modal
from hypernodes.backend import ModalBackend

# ==================== CONFIGURATION ====================
# Paths - adjust if needed
hypernodes_dir = Path("/Users/giladrubin/python_workspace/hypernodes")

# Create volumes
models_volume = modal.Volume.from_name("mafat-models", create_if_missing=True)
data_volume = modal.Volume.from_name("mafat-data", create_if_missing=True)

# ==================== FIXED IMAGE DEFINITION ====================
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({
        "HF_HOME": "/root/models",
        "PYTHONUNBUFFERED": "1",
        # KEY FIX: Add hypernodes to Python path
        "PYTHONPATH": "/root/hypernodes/src:$PYTHONPATH",
    })
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
    # KEY FIX: Copy src folder, not entire repo
    .add_local_dir(
        str(hypernodes_dir / "src"),
        remote_path="/root/hypernodes/src",
    )
)

# ==================== FIXED BACKEND DEFINITION ====================
modal_backend = ModalBackend(
    image=image,
    gpu="A10G",
    # KEY FIX: Longer timeout for long-running pipeline
    timeout=7200,  # 2 hours - adjust based on your needs
    map_execution="sequential",
    volumes={
        "/root/models": models_volume,
        "/root/data": data_volume,
    },
)

print("✓ Modal backend configured with fixes:")
print("  - Hypernodes added to PYTHONPATH")
print("  - Timeout increased to 2 hours")
print("  - Proper src folder mounting")
print("\nYou can now use: pipeline = pipeline.with_engine(modal_backend)")
