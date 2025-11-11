"""Custom models module - simulates having classes in a separate file.

This module contains classes that will be imported by the main script,
triggering the module resolution issue in DaftEngine.
"""


class TextData:
    """A simple data class to demonstrate the serialization issue."""

    def __init__(self, content: str):
        self.content = content

    def __repr__(self):
        return f"TextData('{self.content}')"


class TextEncoder:
    """Encoder that processes TextData objects - method explicitly typed."""

    def __init__(self, prefix: str = "ENCODED"):
        self.prefix = prefix

    def encode(self, data: TextData) -> TextData:
        """
        Encode TextData - THIS METHOD SIGNATURE IS THE KEY!
        The type annotation references TextData from this module (custom_models).
        When this gets pickled by cloudpickle for Daft's UDF worker,
        the worker process needs to be able to import 'custom_models.TextData'.
        """
        return TextData(f"{self.prefix}[{data.content}]")
