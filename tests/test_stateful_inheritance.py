"""Test case: @stateful decorator partially breaks inheritance.

The `@stateful` decorator wraps a class in a `StatefulWrapper` that uses
lazy initialization. There are subtle issues with method overrides:

WHAT WORKS:
    - Direct method calls to overridden methods work
    - The wrapper's __getattr__ finds subclass methods first

WHAT DOESN'T WORK (THE BUG):
    - When a BASE class method calls another method that's overridden,
      the override is NOT called - it calls the base class version
    - This is because self._instance is the ORIGINAL class, so when
      the base method does self.helper(), it's calling the base helper

EXAMPLE:
    class DoubleProcessor(BaseProcessor):
        def get_multiplier(self) -> int:
            return 2  # Override

    processor = DoubleProcessor()
    processor.get_multiplier()  # Returns 2 ✓ (direct call works)
    processor.compute(5)  # Returns 5 ✗ (compute calls self.get_multiplier
                          # but gets base class version, returns 1)

This makes it impossible to customize @stateful classes through inheritance
when the customization involves methods called by other base class methods,
which is a common pattern for testing (creating mock subclasses).
"""

import pytest
from hypernodes import stateful


@stateful
class BaseProcessor:
    """Example stateful class."""

    def __init__(self, prefix: str = "base"):
        self.prefix = prefix
        self.call_count = 0

    def process(self, value: str) -> str:
        """Process a value - should be overridable."""
        self.call_count += 1
        return f"{self.prefix}: {value}"

    def get_multiplier(self) -> int:
        """Helper method - should be overridable."""
        return 1

    def compute(self, x: int) -> int:
        """Uses get_multiplier - to test that overrides chain correctly."""
        return x * self.get_multiplier()


class TestStatefulInheritance:
    """Tests for @stateful inheritance issues."""

    def test_basic_stateful_works(self):
        """Verify basic @stateful functionality works."""
        processor = BaseProcessor(prefix="test")
        result = processor.process("hello")
        assert result == "test: hello"
        assert processor.call_count == 1

    def test_direct_method_override_works(self):
        """Direct method override DOES work (surprisingly).
        
        When you call an overridden method directly, it works because
        __getattr__ finds the method on the subclass wrapper first.
        """

        class CustomProcessor(BaseProcessor):
            def process(self, value: str) -> str:
                self.call_count += 1
                return f"CUSTOM: {value.upper()}"

        processor = CustomProcessor(prefix="ignored")
        result = processor.process("hello")

        # This actually WORKS - direct method calls find the subclass method
        assert result == "CUSTOM: HELLO", f"Got '{result}'"

    def test_direct_helper_call_works(self):
        """Direct call to overridden helper method works."""

        class DoubleProcessor(BaseProcessor):
            def get_multiplier(self) -> int:
                return 2  # Override to double

        processor = DoubleProcessor()
        result = processor.get_multiplier()

        # Direct call WORKS
        assert result == 2, f"Got {result}"

    def test_chained_method_call_fails(self):
        """When base method calls overridden helper, override is NOT used.
        
        THIS IS THE BUG: compute() calls self.get_multiplier(), but since
        self._instance is BaseProcessor, it calls BaseProcessor.get_multiplier()
        not DoubleProcessor.get_multiplier().
        """

        class DoubleProcessor(BaseProcessor):
            def get_multiplier(self) -> int:
                return 2  # Override to double

        processor = DoubleProcessor()

        # Direct call works
        assert processor.get_multiplier() == 2, "Direct call should work"

        # Chained call through compute() FAILS
        result = processor.compute(5)

        # EXPECTED: 10 (5 * 2 from override)
        # ACTUAL: 5 (5 * 1 from base class)
        assert result == 10, f"Got {result} - chained override was not called!"

    def test_subclass_init_override_should_work(self):
        """Subclass __init__ override should be called.
        
        CURRENTLY FAILS: Subclass __init__ is never called.
        """

        class InitOverrideProcessor(BaseProcessor):
            def __init__(self, prefix: str = "base"):
                super().__init__(prefix)
                self.extra_field = "I was initialized by subclass"

        processor = InitOverrideProcessor(prefix="test")

        # Force initialization by accessing a method
        _ = processor.process("test")

        # EXPECTED: extra_field exists
        # ACTUAL: AttributeError because subclass __init__ was never called
        assert hasattr(processor, "extra_field"), "Subclass __init__ was not called!"
        assert processor.extra_field == "I was initialized by subclass"

    def test_isinstance_check_fails(self):
        """isinstance checks don't work with @stateful wrapper.
        
        This is a secondary issue - the wrapper class is not recognized
        as an instance of the original class.
        """

        processor = BaseProcessor()

        # This might fail depending on implementation
        # The wrapper should ideally pass isinstance checks
        # assert isinstance(processor, BaseProcessor)  # May or may not work

    def test_type_check_fails(self):
        """type() returns wrapper class, not original class."""
        processor = BaseProcessor()

        # type() returns StatefulWrapper, not BaseProcessor
        # This can cause issues with type-based dispatch
        actual_type = type(processor).__name__

        # Note: This is expected behavior for the wrapper pattern,
        # but can be surprising for users
        assert actual_type == "BaseProcessor" or actual_type == "StatefulWrapper"


class TestStatefulInheritanceWorkaround:
    """Workaround examples for the inheritance issue."""

    def test_workaround_composition(self):
        """Workaround: Use composition instead of inheritance."""

        class CustomProcessor:
            """Custom processor using composition."""

            def __init__(self, prefix: str = "custom"):
                self.prefix = prefix
                self.call_count = 0

            def process(self, value: str) -> str:
                self.call_count += 1
                return f"CUSTOM: {value.upper()}"

        processor = CustomProcessor()
        result = processor.process("hello")
        assert result == "CUSTOM: HELLO"

    def test_workaround_no_stateful_for_testing(self):
        """Workaround: Don't use @stateful for classes that need subclassing."""

        class PlainProcessor:
            """Plain class without @stateful - can be subclassed."""

            def __init__(self, prefix: str = "base"):
                self.prefix = prefix

            def process(self, value: str) -> str:
                return f"{self.prefix}: {value}"

        class CustomPlainProcessor(PlainProcessor):
            def process(self, value: str) -> str:
                return f"CUSTOM: {value.upper()}"

        processor = CustomPlainProcessor()
        result = processor.process("hello")
        assert result == "CUSTOM: HELLO"


# === Reproduction Script ===


def demonstrate_bug():
    """Standalone demonstration of the @stateful inheritance bug."""
    print("=" * 60)
    print("@stateful Inheritance Bug Demonstration")
    print("=" * 60)

    # Define base class
    @stateful
    class MockLLM:
        def __init__(self, default_response: str = "default"):
            self.default_response = default_response

        def _get_response(self, prompt: str) -> str:
            """Helper method - should be overridable."""
            return self.default_response

        def generate(self, prompt: str) -> str:
            """Public method that calls _get_response."""
            return self._get_response(prompt)

    # Try to subclass and override
    class CustomMockLLM(MockLLM):
        def _get_response(self, prompt: str) -> str:
            return "CUSTOM RESPONSE"

    print("\n1. Base class works:")
    base = MockLLM(default_response="base response")
    print(f"   base.generate('test') = '{base.generate('test')}'")
    print(f"   ✓ Expected: 'base response'")

    print("\n2. Direct override call WORKS:")
    custom = CustomMockLLM(default_response="ignored")
    direct_result = custom._get_response("test")
    print(f"   custom._get_response('test') = '{direct_result}'")
    if direct_result == "CUSTOM RESPONSE":
        print("   ✓ Direct call to overridden method works!")
    else:
        print(f"   ✗ Got '{direct_result}'")

    print("\n3. Chained call through base method FAILS:")
    chained_result = custom.generate("test")
    print(f"   custom.generate('test') = '{chained_result}'")
    print(f"   Expected: 'CUSTOM RESPONSE'")
    print(f"   Actual: '{chained_result}'")

    if chained_result == "CUSTOM RESPONSE":
        print("   ✓ Override worked!")
    else:
        print("   ✗ Override was NOT called - this is the bug!")

    print("\n4. Why this happens:")
    print("   - @stateful wraps MockLLM in StatefulWrapper")
    print("   - CustomMockLLM subclasses StatefulWrapper")
    print("   - Direct method calls (custom._get_response) find subclass method ✓")
    print("   - BUT generate() is on self._instance (original MockLLM)")
    print("   - When generate() calls self._get_response(), 'self' is MockLLM")
    print("   - So it calls MockLLM._get_response, not the override ✗")

    print("\n5. Impact:")
    print("   - Cannot override helper methods called by base class methods")
    print("   - Common pattern: MockLLM.generate() calls self._get_raw_response()")
    print("   - Subclass override of _get_raw_response() is ignored")
    print("   - Workaround: Use composition or override the top-level method")


if __name__ == "__main__":
    demonstrate_bug()

