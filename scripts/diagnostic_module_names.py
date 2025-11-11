#!/usr/bin/env python3
"""Diagnostic script to understand module naming differences.

Run this from different directories to see how __module__ changes:
1. From repo root: uv run scripts/diagnostic_module_names.py
2. From scripts/: cd scripts && uv run diagnostic_module_names.py
"""

from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel
import sys


# Define classes similar to test_modal_failure_repro.py
class Passage(BaseModel):
    uuid: str
    text: str
    model_config = {"frozen": True}


class Prediction(BaseModel):
    query_uuid: str
    paragraph_uuid: str
    score: float
    model_config = {"frozen": True}


class RecallEvaluator:
    """Intentionally missing __daft_hint__ (matches failing scenario)."""

    def __init__(self, k_list: List[int]):
        self.k_list = k_list

    def compute(self, predictions: List[Prediction], ground_truths: List) -> Dict[str, float]:
        metrics = {}
        for k in self.k_list:
            metrics[f"recall@{k}"] = min(len(predictions), k)
        return metrics


def main():
    print("=" * 80)
    print("MODULE NAME DIAGNOSTICS")
    print("=" * 80)
    print()
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Script __name__: {__name__}")
    print(f"Script __file__: {__file__}")
    print()
    print("-" * 80)
    print("CLASS MODULES:")
    print("-" * 80)
    print(f"Passage.__module__ = {Passage.__module__!r}")
    print(f"Prediction.__module__ = {Prediction.__module__!r}")
    print(f"RecallEvaluator.__module__ = {RecallEvaluator.__module__!r}")
    print()
    
    # Create instances and check their module
    passage = Passage(uuid="p1", text="test")
    prediction = Prediction(query_uuid="q1", paragraph_uuid="p1", score=0.5)
    evaluator = RecallEvaluator(k_list=[1, 2])
    
    print("-" * 80)
    print("INSTANCE CLASS MODULES:")
    print("-" * 80)
    print(f"passage.__class__.__module__ = {passage.__class__.__module__!r}")
    print(f"prediction.__class__.__module__ = {prediction.__class__.__module__!r}")
    print(f"evaluator.__class__.__module__ = {evaluator.__class__.__module__!r}")
    print()
    
    # Check if modules are importable
    print("-" * 80)
    print("MODULE IMPORTABILITY:")
    print("-" * 80)
    for mod_name in [Passage.__module__, RecallEvaluator.__module__]:
        if mod_name in sys.modules:
            print(f"✓ {mod_name!r} is in sys.modules")
        else:
            print(f"✗ {mod_name!r} is NOT in sys.modules")
            try:
                __import__(mod_name)
                print(f"  └─ But it CAN be imported")
            except (ImportError, ModuleNotFoundError) as e:
                print(f"  └─ And it CANNOT be imported: {e}")
    print()
    
    # Test cloudpickle serialization
    print("-" * 80)
    print("CLOUDPICKLE SERIALIZATION TEST:")
    print("-" * 80)
    try:
        import cloudpickle
        
        # Try to pickle the evaluator (stateful object)
        try:
            pickled = cloudpickle.dumps(evaluator)
            unpickled = cloudpickle.loads(pickled)
            print(f"✓ RecallEvaluator can be pickled/unpickled")
            print(f"  └─ Unpickled module: {unpickled.__class__.__module__!r}")
        except Exception as e:
            print(f"✗ RecallEvaluator CANNOT be pickled: {e}")
        
        # Try to pickle Pydantic instances
        try:
            pickled = cloudpickle.dumps(prediction)
            unpickled = cloudpickle.loads(pickled)
            print(f"✓ Prediction can be pickled/unpickled")
            print(f"  └─ Unpickled module: {unpickled.__class__.__module__!r}")
        except Exception as e:
            print(f"✗ Prediction CANNOT be pickled: {e}")
            
    except ImportError:
        print("cloudpickle not available, skipping serialization test")
    
    print()
    print("=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
