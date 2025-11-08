"""
Profile each component of the pipeline execution to find remaining bottlenecks.
"""

import asyncio
import cProfile
import io
import pstats
import time

from hypernodes import HypernodesEngine, Pipeline, node
from hypernodes.executors import AsyncExecutor


# Native async function
@node(output_name="async_result")
async def native_async_fn(delay: float) -> dict:
    await asyncio.sleep(delay)
    return {"delay": delay}


delays = [0.1] * 50

print("=" * 70)
print("🔬 PROFILING HYPERNODES EXECUTION")
print("=" * 70)

executor = AsyncExecutor(max_workers=50)
pipeline = Pipeline(
    nodes=[native_async_fn],
    backend=HypernodesEngine(map_executor=executor),
)

# Profile the execution
pr = cProfile.Profile()
pr.enable()

start = time.time()
results = pipeline.map(inputs={"delay": delays}, map_over="delay")
elapsed = time.time() - start

pr.disable()

print(f"\nTotal time: {elapsed:.3f}s")
print("\nTop 20 slowest functions:\n")

# Print stats
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(20)
print(s.getvalue())

executor.shutdown()
