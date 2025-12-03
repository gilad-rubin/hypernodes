"""Test case: @stateful doesn't patch single-underscore method overrides.

The current fix for @stateful inheritance skips ALL methods starting with `_`,
but it should only skip dunder methods (`__`). Single-underscore methods like
`_get_raw_response` are commonly used as "protected" methods meant to be
overridden by subclasses.

CURRENT BEHAVIOR (BUG):
    - Methods starting with `_` are skipped during patching (line 90-91 in decorators.py)
    - This prevents subclass overrides of `_helper()` methods from being called
    - Only dunder methods (`__init__`, `__call__`, etc.) should be skipped

EXPECTED BEHAVIOR:
    - Single-underscore methods like `_helper()` should be patchable
    - Only double-underscore (dunder) methods should be skipped

LOCATION OF BUG:
    decorators.py, _patch_overridden_methods(), lines 90-91:

    for name in wrapper_type.__dict__:
        if name.startswith('_'):  # <-- Should be name.startswith('__')
            continue

FIX:
    Change line 91 from:
        if name.startswith('_'):
    To:
        if name.startswith('__'):
"""

from hypernodes import stateful


@stateful
class BaseProcessor:
    """Example stateful class with underscore helper method."""

    def __init__(self, prefix: str = "base"):
        self.prefix = prefix

    def _get_response(self, value: str) -> str:
        """Protected helper method - should be overridable by subclasses."""
        return f"{self.prefix}: {value}"

    def process(self, value: str) -> str:
        """Public method that uses _get_response internally."""
        return self._get_response(value)


class TestStatefulUnderscoreMethods:
    """Tests for single-underscore method override support."""

    def test_base_class_works(self):
        """Verify base class works correctly."""
        processor = BaseProcessor(prefix="test")
        assert processor.process("hello") == "test: hello"

    def test_direct_underscore_method_call_works(self):
        """Direct call to overridden underscore method works."""

        class CustomProcessor(BaseProcessor):
            def _get_response(self, value: str) -> str:
                return f"CUSTOM: {value.upper()}"

        processor = CustomProcessor()

        # Direct call works (found on wrapper subclass)
        result = processor._get_response("hello")
        assert result == "CUSTOM: HELLO", f"Direct call failed: got '{result}'"

    def test_chained_underscore_method_override_fails(self):
        """Chained call through base method should use override.

        THIS IS THE BUG: process() calls self._get_response(), but the
        override is not patched because it starts with `_`.
        """

        class CustomProcessor(BaseProcessor):
            def _get_response(self, value: str) -> str:
                return f"CUSTOM: {value.upper()}"

        processor = CustomProcessor()

        # Direct call works
        assert processor._get_response("hello") == "CUSTOM: HELLO"

        # Chained call through process() SHOULD use override
        result = processor.process("hello")

        # EXPECTED: "CUSTOM: HELLO" (from override)
        # ACTUAL: "base: hello" (from base class, because _get_response wasn't patched)
        assert result == "CUSTOM: HELLO", (
            f"Chained call failed: got '{result}'. "
            f"The _get_response override was not patched!"
        )


class TestRealWorldMockLLMPattern:
    """Real-world pattern: MockLLM with overridable _get_raw_response."""

    def test_mock_llm_override_pattern(self):
        """Common testing pattern: override _get_raw_response in MockLLM subclass."""

        @stateful
        class MockLLM:
            def __init__(self, default_response: str = "default"):
                self.default_response = default_response

            def _get_raw_response(self, prompt: str) -> str:
                """Override this in subclasses to customize mock responses."""
                if "valid" in prompt.lower():
                    return '{"is_valid": true}'
                return self.default_response

            def generate(self, prompt: str) -> str:
                """Generate response using _get_raw_response."""
                return self._get_raw_response(prompt)

        class InvalidQueryMockLLM(MockLLM):
            """Subclass that always returns invalid for testing rejection paths."""

            def _get_raw_response(self, prompt: str) -> str:
                if "valid" in prompt.lower():
                    return '{"is_valid": false, "reason": "Test rejection"}'
                return super()._get_raw_response(prompt)

        # Test base class
        base_llm = MockLLM()
        assert base_llm.generate("Is this valid?") == '{"is_valid": true}'

        # Test subclass override
        custom_llm = InvalidQueryMockLLM()

        # Direct call works
        direct = custom_llm._get_raw_response("Is this valid?")
        assert '"is_valid": false' in direct, f"Direct call failed: {direct}"

        # Chained call through generate() SHOULD use override
        result = custom_llm.generate("Is this valid?")

        # EXPECTED: '{"is_valid": false, "reason": "Test rejection"}'
        # ACTUAL: '{"is_valid": true}' (override not called)
        assert '"is_valid": false' in result, (
            f"generate() didn't use override: got '{result}'. "
            f"The _get_raw_response override was not patched!"
        )


# === Standalone Demonstration ===


def demonstrate_bug():
    """Standalone demonstration of the underscore method bug."""
    print("=" * 70)
    print("@stateful Bug: Single-underscore methods not patched for subclasses")
    print("=" * 70)

    @stateful
    class MockLLM:
        def __init__(self, default: str = "default"):
            self.default = default

        def _get_response(self, prompt: str) -> str:
            """Protected method - should be overridable."""
            return self.default

        def generate(self, prompt: str) -> str:
            """Public method that calls _get_response."""
            return self._get_response(prompt)

    class CustomMockLLM(MockLLM):
        def _get_response(self, prompt: str) -> str:
            return "CUSTOM OVERRIDE"

    print("\n1. Base class works:")
    base = MockLLM(default="base response")
    print(f"   base.generate('test') = '{base.generate('test')}'")
    print("   ✓ Expected: 'base response'")

    print("\n2. Direct call to override WORKS:")
    custom = CustomMockLLM()
    direct = custom._get_response("test")
    print(f"   custom._get_response('test') = '{direct}'")
    if direct == "CUSTOM OVERRIDE":
        print("   ✓ Direct call works")
    else:
        print("   ✗ Direct call failed: expected 'CUSTOM OVERRIDE'")

    print("\n3. Chained call through generate() FAILS:")
    chained = custom.generate("test")
    print(f"   custom.generate('test') = '{chained}'")
    print("   Expected: 'CUSTOM OVERRIDE'")

    if chained == "CUSTOM OVERRIDE":
        print("   ✓ Chained call works - BUG IS FIXED!")
    else:
        print("   ✗ Chained call failed - override not patched!")
        print("\n" + "=" * 70)
        print("ROOT CAUSE:")
        print("=" * 70)
        print("""
In decorators.py, _patch_overridden_methods() skips ALL methods starting with '_':

    for name in wrapper_type.__dict__:
        if name.startswith('_'):    # <-- BUG: skips _get_response
            continue

This should only skip dunder methods:

    for name in wrapper_type.__dict__:
        if name.startswith('__'):   # <-- FIX: only skip __init__, __call__, etc.
            continue
""")

    print("\n" + "=" * 70)
    print("IMPACT:")
    print("=" * 70)
    print("""
- Cannot override protected methods in @stateful class subclasses
- Common pattern: MockLLM._get_raw_response() for test customization
- Workaround: Rename to public method (no underscore) - but breaks convention
""")


if __name__ == "__main__":
    demonstrate_bug()
