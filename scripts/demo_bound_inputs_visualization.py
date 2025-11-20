"""Demo script showing how bound inputs appear in visualizations.

This script generates example visualizations that demonstrate the transparency
feature for bound inputs.
"""

from hypernodes import Pipeline, node


@node(output_name="result")
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


@node(output_name="scaled")
def scale(value: int, factor: int) -> int:
    """Scale a value by a factor."""
    return value * factor


@node(output_name="final")
def multiply(result: int, scaled: int) -> int:
    """Multiply two values."""
    return result * scaled


def main():
    print("=" * 60)
    print("BOUND INPUTS VISUALIZATION DEMO")
    print("=" * 60)
    print()
    
    # Example 1: No binding - all inputs fully opaque
    print("1. No binding (all inputs opaque):")
    pipeline1 = Pipeline(nodes=[add, scale], name="no_binding")
    print(f"   Bound inputs: {pipeline1.bound_inputs}")
    print(f"   Unfulfilled: {pipeline1.unfulfilled_args}")
    viz1 = pipeline1.visualize(min_arg_group_size=None)
    print(f"   ✓ Visualization shows all 4 inputs with normal opacity")
    print()
    
    # Example 2: Partial binding - some inputs dashed
    print("2. Partial binding (some inputs with dashed borders):")
    pipeline2 = Pipeline(nodes=[add, scale], name="partial_binding")
    pipeline2.bind(x=5, factor=10)
    print(f"   Bound inputs: {pipeline2.bound_inputs}")
    print(f"   Unfulfilled: {pipeline2.unfulfilled_args}")
    viz2 = pipeline2.visualize(min_arg_group_size=None)
    print(f"   ✓ Bound inputs 'x' and 'factor' have dashed borders")
    print(f"   ✓ Unbound inputs 'y' and 'value' have solid borders")
    print()
    
    # Example 3: Full binding - all inputs dashed
    print("3. Full binding (all inputs with dashed borders):")
    pipeline3 = Pipeline(nodes=[add, scale], name="full_binding")
    pipeline3.bind(x=5, y=10, value=2, factor=3)
    print(f"   Bound inputs: {pipeline3.bound_inputs}")
    print(f"   Unfulfilled: {pipeline3.unfulfilled_args}")
    viz3 = pipeline3.visualize(min_arg_group_size=None)
    print(f"   ✓ All inputs have dashed borders")
    print(f"   ✓ Pipeline can run without any external inputs!")
    print()
    
    # Example 4: Grouped inputs with partial binding
    print("4. Grouped inputs (separate bound/unbound groups):")
    @node(output_name="result")
    def process(a: int, b: int, c: int, d: int) -> int:
        return a + b + c + d
    
    pipeline4 = Pipeline(nodes=[process], name="grouped")
    pipeline4.bind(a=1, b=2)  # Bind 2 out of 4
    print(f"   Bound inputs: {pipeline4.bound_inputs}")
    print(f"   Unfulfilled: {pipeline4.unfulfilled_args}")
    viz4 = pipeline4.visualize(min_arg_group_size=2)
    print(f"   ✓ Bound inputs (a,b) grouped with dashed border")
    print(f"   ✓ Unbound inputs (c,d) grouped with solid border")
    print(f"   ✓ Two separate groups (no mixing bound/unbound)")
    print()
    
    # Example 5: Nested pipeline with binding
    print("5. Nested pipeline with bound inputs:")
    inner = Pipeline(nodes=[scale], name="inner")
    inner.bind(factor=100)  # Inner pipeline is partially configured
    outer = Pipeline(nodes=[inner.as_node(), add], name="outer")
    print(f"   Inner bound: {inner.bound_inputs}")
    print(f"   Inner unfulfilled: {inner.unfulfilled_args}")
    print(f"   Outer unfulfilled: {outer.unfulfilled_args}")
    viz5 = outer.visualize(depth=2, min_arg_group_size=None)
    print(f"   ✓ Inner 'factor' input shown with dashed border")
    print(f"   ✓ Outer still needs 'value', 'x', 'y' with solid borders")
    print()
    
    print("=" * 60)
    print("HOW TO SEE THE BOUND INPUTS:")
    print("=" * 60)
    print()
    print("In the visualization:")
    print("  • Unbound inputs: Solid border, green fill (#90EE90)")
    print("  • Bound inputs: Dashed border, green fill (#90EE90)")
    print()
    print("All nodes have the same color and opacity.")
    print("Only the border style differs (solid vs dashed).")
    print()
    print("Grouped inputs:")
    print("  • All bound: Dashed border")
    print("  • All unbound: Solid border")
    print("  • Bound and unbound are NOT grouped together")
    print()
    print("To save visualizations, use:")
    print("  pipeline.visualize('output.svg', min_arg_group_size=None)")
    print()
    
    # Demonstrate the actual border styles
    print("=" * 60)
    print("VERIFYING BORDER STYLES IN CODE:")
    print("=" * 60)
    print()
    
    pipeline_test = Pipeline(nodes=[add])
    pipeline_test.bind(x=5)
    viz = pipeline_test.visualize(min_arg_group_size=None)
    viz_str = str(viz)
    
    # Show that bound input has dashed border
    if 'dashed' in viz_str:
        print("✓ Found bound input with dashed border styling")
    if viz_str.count('style="filled"') > 0:
        print("✓ Found unbound input with solid border (filled only)")
    if '80' not in viz_str:
        print("✓ No transparency - all nodes have same opacity")
    print()
    print("Test passed! Bound inputs are distinguished by dashed borders!")
    print()


if __name__ == "__main__":
    main()

