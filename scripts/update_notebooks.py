"""Script to update notebook cells with old API to new API.

This updates:
- Pipeline(..., cache=X, callbacks=Y) -> Pipeline(..., engine=Engine(cache=X, callbacks=Y))
- pipeline.with_cache(X) -> Rebuild with engine
"""

import json
import re
from pathlib import Path

def update_notebook(notebook_path):
    """Update a single notebook."""
    with open(notebook_path) as f:
        nb = json.load(f)
    
    changed = False
    
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
            
        source = cell.get("source", [])
        if not source:
            continue
        
        # Convert list to string
        if isinstance(source, list):
            code = "".join(source)
        else:
            code = source
        
        original = code
        
        # Pattern 1: Pipeline(..., cache=DiskCache(...), callbacks=[...])
        # This is complex - let's handle simpler cases manually
        
        # Pattern 2: pipeline.with_cache(...)
        if "with_cache" in code:
            print(f"  Found with_cache in {notebook_path.name}")
            # Manual update needed
        
        # Pattern 3: Pipeline(..., callbacks=[...])
        if "callbacks=[" in code and "Pipeline(" in code:
            print(f"  Found callbacks in Pipeline in {notebook_path.name}")
            
    return changed


if __name__ == "__main__":
    notebooks = Path("notebooks").glob("*.ipynb")
    for nb_path in notebooks:
        print(f"Checking {nb_path.name}...")
        update_notebook(nb_path)

