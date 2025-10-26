"""Telemetry and observability for HyperNodes pipelines.

Provides:
- ProgressCallback: Live progress bars (tqdm/rich)
- TelemetryCallback: Distributed tracing with Logfire
- Waterfall charts for post-hoc analysis (Jupyter only)
"""

from .progress import ProgressCallback
from .tracing import TelemetryCallback

__all__ = [
    "ProgressCallback",
    "TelemetryCallback",
]
