"""Debug progress bar updates."""
import time

from hypernodes import Pipeline, node
from hypernodes.telemetry import ProgressCallback


@node(output_name="step1")
def step_one(x: int) -> int:
    """Step 1."""
    time.sleep(0.5)
    return x * 2


@node(output_name="step2")
def step_two(step1: int) -> int:
    """Step 2."""
    time.sleep(0.5)
    return step1 + 1


@node(output_name="step3")
def step_three(step2: int) -> int:
    """Step 3."""
    time.sleep(0.5)
    return step2 * 3


# Custom progress callback with debugging
class DebugProgressCallback(ProgressCallback):
    """Progress callback with debug output."""
    
    def on_node_start(self, node_id: str, inputs, ctx):
        """Debug node start."""
        print(f"\n[DEBUG] Node starting: {node_id}")
        print(f"[DEBUG] Pipeline ID: {ctx.current_pipeline_id}")
        print(f"[DEBUG] In map: {ctx.get('_in_map', False)}")
        result = super().on_node_start(node_id, inputs, ctx)
        
        # Check if bar was updated
        pipeline_bar = ctx.get(f"progress_bar:{ctx.current_pipeline_id}")
        if pipeline_bar:
            print(f"[DEBUG] Pipeline bar description: {pipeline_bar.desc}")
        
        return result


pipeline = Pipeline(
    nodes=[step_one, step_two, step_three],
    callbacks=[DebugProgressCallback()],
    name="test_pipeline",
)

print("Running pipeline with debug output...")
results = pipeline.run(inputs={"x": 5})
print(f"\nFinal result: {results}")
