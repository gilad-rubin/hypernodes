"""Environment detection utilities for telemetry."""


def is_jupyter() -> bool:
    """Detect if running in Jupyter notebook or IPython.
    
    Returns:
        True if running in Jupyter/IPython environment, False otherwise
    """
    try:
        # Check if we're in IPython/Jupyter
        get_ipython  # type: ignore[name-defined]
        return True
    except NameError:
        return False
