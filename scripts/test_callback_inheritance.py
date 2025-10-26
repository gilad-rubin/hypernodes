"""Test callback inheritance."""
from hypernodes import Pipeline, node
from hypernodes.telemetry import ProgressCallback


@node(output_name="x")
def dummy(item: str) -> str:
    return item


# Inner pipeline - no callbacks set
inner = Pipeline(nodes=[dummy], name="inner")

# Check effective callbacks
print(f"inner._parent: {inner._parent}")
print(f"inner.callbacks: {inner.callbacks}")
print(f"inner.effective_callbacks: {inner.effective_callbacks}")

# Wrap it
mapped = inner.as_node(
    input_mapping={"items": "item"},
    output_mapping={"x": "results"},
    map_over="items",
    name="mapped",
)

# Outer pipeline - has callbacks
outer = Pipeline(
    nodes=[mapped],
    callbacks=[ProgressCallback()],
    name="outer",
)

print(f"\nouter.callbacks: {outer.callbacks}")
print(f"outer.effective_callbacks: {outer.effective_callbacks}")

# The issue: inner.effective_callbacks is still [] because it doesn't have a parent!
print(f"\nAfter wrapping:")
print(f"inner._parent: {inner._parent}")
print(f"inner.effective_callbacks: {inner.effective_callbacks}")
