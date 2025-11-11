"""Minimal script to reproduce DaftEngine serialization bug with custom classes.

This script demonstrates the issue where custom classes defined in a script
fail to deserialize in Daft's UDF worker process, even after module name fixing.
"""

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# Simple custom class that will be used as a stateful parameter
class TextProcessor:
    """A simple processor to demonstrate the serialization issue."""

    def __init__(self, prefix: str = "PROCESSED"):
        self.prefix = prefix

    def process(self, text: str) -> str:
        """Process text by adding a prefix."""
        return f"{self.prefix}: {text}"


# Node that uses the custom class as a stateful parameter
@node(output_name="processed")
def process_text(text: str, processor: TextProcessor) -> str:
    """Process text using the stateful processor."""
    return processor.process(text)


def main():
    """Run the minimal test case."""
    print("Creating TextProcessor instance...")
    processor = TextProcessor(prefix="TEST")

    print("Building pipeline...")
    pipeline = Pipeline(
        nodes=[process_text],
        engine=DaftEngine(collect=True),
    )

    print("Running pipeline with DaftEngine...")
    inputs = {
        "text": "Hello World",
        "processor": processor,  # Stateful parameter
    }

    try:
        result = pipeline.run(inputs=inputs)
        print(f"Success! Result: {result}")
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}")
        print(f"Error message: {e}")
        raise


if __name__ == "__main__":
    main()
