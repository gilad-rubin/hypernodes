"""
Test to confirm AsyncExecutor bottleneck with native async functions.
"""

import asyncio
import time

from hypernodes import HypernodesEngine, Pipeline, node
from hypernodes.executors import AsyncExecutor


# Native async function
@node(output_name="async_result")
async def native_async_fn(delay: float) -> dict:
    await asyncio.sleep(delay)  # Non-blocking async sleep
    return {"delay": delay, "completed": time.time()}


# Test parameters
delays = [0.1] * 50

print("=" * 70)
print("🧪 HYPERNODES ASYNC EXECUTOR TEST")
print("=" * 70)

# Test 1: HyperNodes with different async strategies
print("\n1️⃣  HyperNodes AsyncExecutor (native async function, multiple strategies)")


def run_pipeline(async_strategy: str) -> float:
    engine = HypernodesEngine(
        map_executor=AsyncExecutor(max_workers=50),
        async_strategy=async_strategy,
    )
    pipeline_hn_async = Pipeline(nodes=[native_async_fn], backend=engine)
    start_time = time.time()
    pipeline_hn_async.map(inputs={"delay": delays}, map_over="delay")
    duration = time.time() - start_time
    engine.shutdown()
    return duration


strategy_labels = [
    ("per_call", "New loop per await (baseline)"),
    ("thread_local", "Thread-local event loop reuse"),
    ("async_native", "Async-native pipeline"),
    ("auto", "Auto-detect (hybrid)"),
]

strategy_results = []
for strategy, label in strategy_labels:
    duration = run_pipeline(strategy)
    strategy_results.append((strategy, label, duration))
    print(f"   {label:<32} {duration:.3f}s ({len(delays) * 0.1 / duration:.2f}x speedup)")

# Test 2: Pure asyncio for comparison
print("\n2️⃣  Pure asyncio.gather (baseline)")


async def async_io_operation(delay: float) -> dict:
    await asyncio.sleep(delay)
    return {"delay": delay, "completed": time.time()}


async def run_pure_async():
    return await asyncio.gather(*[async_io_operation(d) for d in delays])


start = time.time()
results_pure = asyncio.run(run_pure_async())
time_pure = time.time() - start
print(f"   Time: {time_pure:.3f}s ({len(delays) * 0.1 / time_pure:.2f}x speedup)")

per_call_time = next(duration for strategy, _, duration in strategy_results if strategy == "per_call")
best_strategy = min(strategy_results, key=lambda item: item[2])

print("\n" + "=" * 70)
print("📊 ANALYSIS")
print("=" * 70)
print(f"Pure asyncio.gather:        {time_pure:.3f}s")
print(f"HyperNodes (per_call):      {per_call_time:.3f}s  ({per_call_time / time_pure:.2f}x slowdown)")
print(
    f"HyperNodes (best: {best_strategy[1]}): "
    f"{best_strategy[2]:.3f}s  ({best_strategy[2] / time_pure:.2f}x slowdown)"
)
print(
    f"\nBest strategy savings vs baseline: "
    f"{(per_call_time - best_strategy[2]) * 1000:.1f}ms"
)

if best_strategy[2] > time_pure * 2:
    print("\n⚠️  WARNING: Even the best strategy is >2x slower than pure asyncio.")
else:
    print("\n✅ Best strategy keeps overhead under 2x relative to pure asyncio.")
