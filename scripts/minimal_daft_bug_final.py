"""Final test - simulates the Modal scenario where module is not importable.

The key insight: DaftEngine only applies the __module__ fix when it detects
that a module is NOT importable. On Modal, script modules like 'test_modal'
are not in the worker's Python path, triggering the fix logic.

We can simulate this by:
1. Defining classes in __main__ (this file)
2. Manually changing their __module__ attribute to a fake module name
3. This makes them appear to be from an unimportable module
4. DaftEngine's _fix_instance_class will try to fix them
"""

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# Define classes directly in this script
class DataItem:
    """Data class - will be made to appear from fake module."""

    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return f"DataItem('{self.value}')"


class Processor:
    """Processor with typed method - will appear from fake module."""

    def __init__(self, prefix: str):
        self.prefix = prefix

    def process(self, item: DataItem) -> DataItem:
        """Process method with DataItem type annotation."""
        return DataItem(f"{self.prefix}[{item.value}]")


# Manually set __module__ to simulate being from an unimportable module
# This mimics what happens on Modal when classes are defined in the script
DataItem.__module__ = "fake_unimportable_module"
Processor.__module__ = "fake_unimportable_module"

# Also need to fix the method's annotations to reference the "fake" module
import types

# Get the process method and update its annotations
original_process = Processor.process


def create_fake_module_process(self, item: DataItem) -> DataItem:
    """Process method with DataItem type annotation from fake module."""
    return DataItem(f"{self.prefix}[{item.value}]")


# Replace with version that has annotations from fake module
Processor.process = create_fake_module_process


# Node using the processor
@node(output_name="result")
def process_item(text: str, processor: Processor) -> str:
    """Process text using processor from 'unimportable' module."""
    item = DataItem(text)
    result = processor.process(item)
    return result.value


# Force use_process to spawn worker
process_item.func.__daft_udf_config__ = {
    "use_process": True,
    "max_concurrency": 1,
}


def main():
    """Run test simulating Modal scenario."""
    print("="*70)
    print("SIMULATING MODAL SCENARIO: Classes from unimportable module")
    print("="*70)

    print("\n1. Class modules:")
    print(f"   DataItem.__module__ = {DataItem.__module__}")
    print(f"   Processor.__module__ = {Processor.__module__}")

    # Try to import - should fail
    print("\n2. Checking if module is importable...")
    import importlib.util
    spec = importlib.util.find_spec("fake_unimportable_module")
    print(f"   find_spec result: {spec}")
    print("   (None means not importable - triggers DaftEngine fix logic)")

    print("\n3. Creating processor instance")
    processor = Processor(prefix="FIXED")

    print("\n4. Building pipeline with DaftEngine")
    pipeline = Pipeline(
        nodes=[process_item],
        engine=DaftEngine(collect=True, debug=True),
    )

    print("\n5. Running with use_process=True")
    print("   DaftEngine will detect module is unimportable")
    print("   It will try to fix the instance class")
    print("   Worker process needs to unpickle the fixed class")
    print()

    try:
        result = pipeline.run(inputs={"text": "Hello", "processor": processor})
        print(f"\n{'='*70}")
        print(f"SUCCESS: {result}")
        print(f"{'='*70}")
        print("\nIf this succeeds, the fix is working correctly!")
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {type(e).__name__}")
        print(f"Message (first 500 chars):")
        print(str(e)[:500])

        if "fake_unimportable_module" in str(e) or "ModuleNotFoundError" in str(e):
            print("\n✓ REPRODUCED THE BUG!")
            print("  Worker tried to import 'fake_unimportable_module' and failed")
            print("  This is exactly what happens on Modal with 'test_modal'")

        print(f"{'='*70}\n")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
