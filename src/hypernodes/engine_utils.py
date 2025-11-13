"""Utility functions for engine operations.

This module provides utility functions used by execution engines.
"""

from contextlib import contextmanager

from .callbacks import CallbackContext
from .execution_context import get_callback_context, set_callback_context


@contextmanager
def managed_callback_context():
    """Context manager for callback context lifecycle.

    Creates a new context if none exists, cleans up on exit.
    Yields existing context if already present.
    """
    ctx = get_callback_context()
    context_created_here = ctx is None

    if context_created_here:
        ctx = CallbackContext()
        set_callback_context(ctx)

    try:
        yield ctx
    finally:
        if context_created_here:
            set_callback_context(None)
