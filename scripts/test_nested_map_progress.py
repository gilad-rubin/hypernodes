"""Test progress with nested mapped pipelines (like retrieval pipeline)."""
import time

from hypernodes import Pipeline, node
from hypernodes.telemetry import ProgressCallback


# Single-item processing
@node(output_name="encoded")
def encode_item(item: str) -> str:
    """Encode single item."""
    time.sleep(0.3)
    return f"encoded_{item}"


# Single-item pipeline
encode_single = Pipeline(
    nodes=[encode_item],
    name="encode_single",
)

# Mapped version using as_node
encode_many = encode_single.as_node(
    input_mapping={"items": "item"},
    output_mapping={"encoded": "encoded_items"},
    map_over="items",
    name="encode_many",
)


# Other nodes
@node(output_name="items")
def load_items() -> list:
    """Load items."""
    return ["a", "b", "c", "d", "e"]


@node(output_name="count")
def count_items(encoded_items: list) -> int:
    """Count items."""
    return len(encoded_items)


# Main pipeline
class DebugProgressCallback(ProgressCallback):
    """Debug progress callback."""
    
    def on_pipeline_start(self, pipeline_id, inputs, ctx):
        print(f"\n[PIPELINE START] {pipeline_id} at depth {ctx.depth}")
        return super().on_pipeline_start(pipeline_id, inputs, ctx)
    
    def on_node_start(self, node_id, inputs, ctx):
        pipeline_id = ctx.current_pipeline_id
        print(f"[NODE START] {node_id} in {pipeline_id} (in_map={ctx.get('_in_map', False)})")
        result = super().on_node_start(node_id, inputs, ctx)
        
        # Check bar description
        bar = ctx.get(f"progress_bar:{pipeline_id}")
        if bar:
            print(f"  → Pipeline bar: '{bar.desc}'")
        
        return result
    
    def on_map_start(self, total_items, ctx):
        print(f"[MAP START] {total_items} items in {ctx.current_pipeline_id}")
        return super().on_map_start(total_items, ctx)


pipeline = Pipeline(
    nodes=[load_items, encode_many, count_items],
    callbacks=[DebugProgressCallback()],
    name="main_pipeline",
)

print("=" * 60)
print("Running pipeline with nested map operations...")
print("=" * 60)

results = pipeline.run(inputs={})

print("\n" + "=" * 60)
print(f"Processed {results['count']} items")
print(f"Results: {results['encoded_items']}")
