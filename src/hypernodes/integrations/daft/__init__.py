"""Daft integration for HyperNodes.

This module provides DaftEngine for distributed DataFrame-based execution.

Example:
    >>> from hypernodes.integrations.daft import DaftEngine
    >>> engine = DaftEngine(collect=True)
    >>> pipeline = Pipeline(nodes=[...], backend=engine)
"""

try:
    from .engine import DaftEngine
    from .engine_v2 import DaftEngineV2
    __all__ = ["DaftEngine", "DaftEngineV2"]
except ImportError:
    # Daft not installed
    __all__ = []
