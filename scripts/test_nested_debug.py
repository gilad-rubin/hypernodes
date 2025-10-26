"""Debug nested map progress."""
import time

from hypernodes import Pipeline, node
from hypernodes.telemetry import ProgressCallback


class DebugProgress(ProgressCallback):
    """Debug progress callbacks."""
    
    def on_map_start(self, total_items, ctx):
        pipeline_id = ctx.current_pipeline_id
        depth = ctx.depth
        print(f"\n[MAP START] pipeline={pipeline_id}, items={total_items}, depth={depth}")
        return super().on_map_start(total_items, ctx)
    
    def on_pipeline_start(self, pipeline_id, inputs, ctx):
        depth = ctx.depth
        print(f"\n[PIPELINE START] {pipeline_id} at depth {depth}")
        return super().on_pipeline_start(pipeline_id, inputs, ctx)


@node(output_name="processed")
def process_item(item: str) -> str:
    time.sleep(0.1)
    return f"proc_{item}"


# Single-item pipeline
process_single = Pipeline(nodes=[process_item], name="process_single")

# Mapped version
process_many = process_single.as_node(
    input_mapping={"items": "item"},
    output_mapping={"processed": "all_processed"},
    map_over="items",
    name="process_many",
)


@node(output_name="items")
def load_items() -> list:
    return ["a", "b", "c", "d", "e"]


# Main pipeline
pipeline = Pipeline(
    nodes=[load_items, process_many],
    callbacks=[DebugProgress()],
    name="main_pipeline",
)

print("=" * 70)
print("Debug: Watch for map_start calls")
print("=" * 70)

results = pipeline.run(inputs={})
print(f"\n✓ Done: {len(results['all_processed'])} items")
