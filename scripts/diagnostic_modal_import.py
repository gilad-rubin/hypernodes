#!/usr/bin/env python3
"""Simulate how Modal imports scripts to diagnose module naming.

Modal's `modal run` doesn't run the script as __main__, it imports it as a module.
This simulates that behavior.
"""

import sys
import os
from pathlib import Path

def test_from_repo_root():
    """Simulate: modal run scripts/test_modal_failure_repro.py (from repo root)"""
    print("=" * 80)
    print("SIMULATING: modal run scripts/test_modal_failure_repro.py")
    print("FROM: repo root")
    print("=" * 80)
    print()
    
    # Modal adds the script's directory to sys.path
    repo_root = Path(__file__).parent.parent
    scripts_dir = repo_root / "scripts"
    
    # When Modal runs scripts/test_modal_failure_repro.py from repo root,
    # it imports as "test_modal_failure_repro" (not scripts.test_modal_failure_repro)
    sys.path.insert(0, str(scripts_dir))
    
    try:
        # This is what Modal does - imports the script as a module
        import test_modal_failure_repro as script_module
        
        print(f"✓ Successfully imported test_modal_failure_repro")
        print()
        print("CLASS MODULE NAMES:")
        print("-" * 80)
        print(f"Passage.__module__ = {script_module.Passage.__module__!r}")
        print(f"Prediction.__module__ = {script_module.Prediction.__module__!r}")
        print(f"RecallEvaluator.__module__ = {script_module.RecallEvaluator.__module__!r}")
        print()
        
        # Check if workers can import this
        print("WORKER IMPORTABILITY:")
        print("-" * 80)
        module_name = script_module.Passage.__module__
        if module_name == "__main__":
            print(f"✓ Module is '__main__' - cloudpickle will serialize by value")
        elif module_name in sys.modules:
            print(f"✓ Module '{module_name}' is in sys.modules")
            print(f"  But will workers have access to this module? Check Modal image!")
        else:
            print(f"✗ Module '{module_name}' NOT in sys.modules")
            print(f"  Workers will crash with: ModuleNotFoundError: No module named '{module_name}'")
        print()
        
        # Test pickling
        print("SERIALIZATION TEST:")
        print("-" * 80)
        try:
            import cloudpickle
            passage = script_module.Passage(uuid="p1", text="test")
            pickled = cloudpickle.dumps(passage)
            print(f"✓ Can pickle Passage instance ({len(pickled)} bytes)")
            
            # This succeeds because we have the module loaded
            unpickled = cloudpickle.loads(pickled)
            print(f"✓ Can unpickle in same process")
            
            # But simulate what happens in a fresh worker
            print()
            print("SIMULATING FRESH WORKER (no module loaded):")
            print("-" * 80)
            # Remove the module
            if 'test_modal_failure_repro' in sys.modules:
                del sys.modules['test_modal_failure_repro']
                print("Removed 'test_modal_failure_repro' from sys.modules")
                
                try:
                    unpickled_in_worker = cloudpickle.loads(pickled)
                    print(f"✓ Worker can unpickle (unexpected!)")
                except ModuleNotFoundError as e:
                    print(f"✗ Worker FAILS: {e}")
                    print(f"  This is the root cause!")
        except ImportError:
            print("cloudpickle not available")
        
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
    finally:
        # Cleanup
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))
        if 'test_modal_failure_repro' in sys.modules:
            del sys.modules['test_modal_failure_repro']


def test_from_scripts_dir():
    """Simulate: cd scripts && modal run test_modal_failure_repro.py"""
    print()
    print("=" * 80)
    print("SIMULATING: modal run test_modal_failure_repro.py")
    print("FROM: scripts/ directory")
    print("=" * 80)
    print()
    
    # Modal adds current directory to sys.path
    repo_root = Path(__file__).parent.parent
    scripts_dir = repo_root / "scripts"
    
    # Change to scripts directory
    original_cwd = os.getcwd()
    os.chdir(scripts_dir)
    sys.path.insert(0, str(scripts_dir))
    
    try:
        # Import the script
        import test_modal_failure_repro as script_module
        
        print(f"✓ Successfully imported test_modal_failure_repro")
        print()
        print("CLASS MODULE NAMES:")
        print("-" * 80)
        print(f"Passage.__module__ = {script_module.Passage.__module__!r}")
        print(f"Prediction.__module__ = {script_module.Prediction.__module__!r}")
        print(f"RecallEvaluator.__module__ = {script_module.RecallEvaluator.__module__!r}")
        print()
        
        module_name = script_module.Passage.__module__
        print("ANALYSIS:")
        print("-" * 80)
        print(f"Module name: {module_name!r}")
        print("Working from scripts/ directory doesn't change module name,")
        print("but it might affect Modal's image setup or PYTHONPATH.")
        print()
        
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
    finally:
        # Cleanup
        os.chdir(original_cwd)
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))
        if 'test_modal_failure_repro' in sys.modules:
            del sys.modules['test_modal_failure_repro']


if __name__ == "__main__":
    test_from_repo_root()
    test_from_scripts_dir()
    
    print()
    print("=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("When Modal imports scripts/test_modal_failure_repro.py, classes get")
    print("__module__ = 'test_modal_failure_repro' (not '__main__')")
    print()
    print("Workers don't have this module, so cloudpickle fails with:")
    print("  ModuleNotFoundError: No module named 'test_modal_failure_repro'")
    print()
    print("SOLUTION: Ensure DaftEngine rewrites __module__ to '__main__' for ALL")
    print("script-defined classes BEFORE they're stored in DataFrames.")
    print("=" * 80)

