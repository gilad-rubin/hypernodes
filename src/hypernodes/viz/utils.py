import os
from pathlib import Path
from typing import Optional

def read_viz_asset(name: str) -> Optional[str]:
    """Read an asset file from assets/viz; return None if missing."""
    try:
        # Viz is at src/hypernodes/viz
        # Assets are at assets/viz
        # We assume this file is in src/hypernodes/viz/utils.py
        # So: utils -> viz -> hypernodes -> src -> root -> assets
        
        # Find the package root (where src/hypernodes is)
        current_file = Path(__file__).resolve()
        
        # If installed as package, assets might be elsewhere. 
        # For now assuming local workspace structure based on previous code:
        # src/hypernodes/viz/js/html_generator.py used .parent.parent.parent.parent.parent
        # from js/html_generator.py (depth 5 from root/src/hypernodes/viz/js ?)
        
        # Let's rely on relative path from this file:
        # src/hypernodes/viz/utils.py
        # -> src/hypernodes/viz (parent)
        # -> src/hypernodes (parent.parent)
        # -> src (parent.parent.parent)
        # -> root (parent.parent.parent.parent)
        
        root = current_file.parent.parent.parent.parent
        path = root / "assets" / "viz" / name
        
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
    except Exception:
        return None

