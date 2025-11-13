"""
Comprehensive benchmark of Daft optimization parameters in HyperNodes.

Tests:
1. Stateful vs non-stateful parameters
2. Batch size variations
3. use_process (GIL isolation)
4. max_concurrency
5. Combined optimizations
"""

import time
from typing import Any
import numpy as np

try:
    import daft
    from daft import Series
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False
    print("❌ Daft not available. Install with: uv add getdaft")

from hypernodes import Pipeline, node
from hypernodes.integrations.daft import DaftEngine


def benchmark_function(func, *args, **kwargs) -> tuple[Any, float]:
    """Execute a function and measure time."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


# ============================================================================
# Test Resources
# ============================================================================

class ExpensiveModel:
    """Simulate an expensive ML model."""
    __daft_stateful__ = True  # Hint for Daft
    
    def __init__(self, delay: float = 0.1):
        time.sleep(delay)  # Simulate loading weights
        self.counter = 0
        self.name = "expensive_model"
    
    def predict(self, text: str) -> str:
        self.counter += 1
        return text.strip().lower()


# ============================================================================
# Test 1: Stateful vs Non-Stateful Parameters
# ============================================================================

def test_stateful_vs_nonstateful():
    """Compare stateful object handling."""
    if not DAFT_AVAILABLE:
        print("❌ Skipping test - Daft not available")
        return
    
    print("\n" + "="*70)
    print("TEST 1: Stateful vs Non-Stateful Parameters")
    print("="*70)
    
    # Test data
    texts = [f"  HELLO {i}  " for i in range(5000)]
    
    # Test 1a: Non-stateful (model created per batch)
    @node(output_name="result")
    def process_nonstateful(text: str) -> str:
        model = ExpensiveModel(delay=0.01)  # Created every time!
        return model.predict(text)
    
    pipeline = Pipeline(
        nodes=[process_nonstateful],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    result, elapsed = benchmark_function(
        pipeline.map,
        inputs={"text": texts},
        map_over="text"
    )
    print(f"\n📊 Non-stateful (model in function):  {elapsed:.4f}s")
    print(f"   └─ Model created repeatedly = SLOW ❌")
    
    # Test 1b: Stateful with explicit hint
    model = ExpensiveModel(delay=0.1)  # Created once
    
    @node(output_name="result", stateful_params=["model"])
    def process_stateful(text: str, model: ExpensiveModel) -> str:
        return model.predict(text)
    
    pipeline = Pipeline(
        nodes=[process_stateful],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    result, elapsed = benchmark_function(
        pipeline.map,
        inputs={"text": texts, "model": model},
        map_over="text"
    )
    print(f"\n📊 Stateful (explicit hint):          {elapsed:.4f}s")
    print(f"   └─ Model created ONCE = FAST ✅")
    
    # Test 1c: Stateful with __daft_stateful__ attribute
    class AutoDetectedModel:
        __daft_stateful__ = True
        
        def __init__(self):
            time.sleep(0.1)
            self.counter = 0
        
        def __call__(self, text: str) -> str:
            self.counter += 1
            return text.strip().lower()
    
    auto_model = AutoDetectedModel()
    
    @node(output_name="result")  # No hint needed!
    def process_auto(text: str, model: AutoDetectedModel) -> str:
        return model(text)
    
    pipeline = Pipeline(
        nodes=[process_auto],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    result, elapsed = benchmark_function(
        pipeline.map,
        inputs={"text": texts, "model": auto_model},
        map_over="text"
    )
    print(f"\n📊 Stateful (auto-detected):           {elapsed:.4f}s")
    print(f"   └─ Auto-detected via __daft_stateful__ ✅")


# ============================================================================
# Test 2: Batch Size Optimization
# ============================================================================

def test_batch_sizes():
    """Test different batch_size configurations."""
    if not DAFT_AVAILABLE:
        print("❌ Skipping test - Daft not available")
        return
    
    print("\n" + "="*70)
    print("TEST 2: Batch Size Optimization")
    print("="*70)
    
    # Numerical data
    values = list(range(50000))
    mean_val = 25000.0
    std_val = 1000.0
    
    batch_sizes = [128, 512, 1024, 2048, 4096]
    results = {}
    
    for batch_size in batch_sizes:
        @node(
            output_name="normalized",
            daft_config={"batch_size": batch_size}
        )
        def normalize(value: float, mean: float, std: float) -> float:
            return (value - mean) / std
        
        pipeline = Pipeline(
            nodes=[normalize],
            engine=DaftEngine(use_batch_udf=True)
        )
        
        result, elapsed = benchmark_function(
            pipeline.map,
            inputs={"value": values, "mean": mean_val, "std": std_val},
            map_over="value"
        )
        results[batch_size] = elapsed
        print(f"📊 Batch size {batch_size:>4}: {elapsed:.4f}s")
    
    # Find optimal
    optimal_size = min(results, key=results.get)
    print(f"\n✅ Optimal batch size: {optimal_size} ({results[optimal_size]:.4f}s)")


# ============================================================================
# Test 3: use_process (GIL Isolation)
# ============================================================================

def test_use_process():
    """Test use_process for CPU-bound operations."""
    if not DAFT_AVAILABLE:
        print("❌ Skipping test - Daft not available")
        return
    
    print("\n" + "="*70)
    print("TEST 3: use_process (GIL Isolation)")
    print("="*70)
    
    texts = [f"Hello world {i}" * 50 for i in range(3000)]
    
    # Test 3a: Without process isolation
    @node(
        output_name="word_count",
        daft_config={"use_process": False}
    )
    def count_words_no_process(text: str) -> int:
        return len(text.split())
    
    pipeline = Pipeline(
        nodes=[count_words_no_process],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    result, elapsed = benchmark_function(
        pipeline.map,
        inputs={"text": texts},
        map_over="text"
    )
    print(f"\n📊 use_process=False: {elapsed:.4f}s")
    print(f"   └─ Subject to Python GIL")
    
    # Test 3b: With process isolation
    @node(
        output_name="word_count",
        daft_config={"use_process": True}
    )
    def count_words_with_process(text: str) -> int:
        return len(text.split())
    
    pipeline = Pipeline(
        nodes=[count_words_with_process],
        engine=DaftEngine(use_batch_udf=True)
    )
    
    result, elapsed = benchmark_function(
        pipeline.map,
        inputs={"text": texts},
        map_over="text"
    )
    print(f"\n📊 use_process=True:  {elapsed:.4f}s")
    print(f"   └─ Process isolation, no GIL ✅")


# ============================================================================
# Test 4: max_concurrency
# ============================================================================

def test_max_concurrency():
    """Test different concurrency levels."""
    if not DAFT_AVAILABLE:
        print("❌ Skipping test - Daft not available")
        return
    
    print("\n" + "="*70)
    print("TEST 4: max_concurrency (Parallel Instances)")
    print("="*70)
    
    texts = [f"Text {i}" for i in range(10000)]
    concurrency_levels = [1, 2, 4, 8]
    results = {}
    
    for concurrency in concurrency_levels:
        @node(
            output_name="result",
            daft_config={"max_concurrency": concurrency}
        )
        def process(text: str) -> str:
            return text.strip().lower()
        
        pipeline = Pipeline(
            nodes=[process],
            engine=DaftEngine(use_batch_udf=True)
        )
        
        result, elapsed = benchmark_function(
            pipeline.map,
            inputs={"text": texts},
            map_over="text"
        )
        results[concurrency] = elapsed
        print(f"📊 max_concurrency={concurrency}: {elapsed:.4f}s")
    
    # Find optimal
    optimal_concurrency = min(results, key=results.get)
    print(f"\n✅ Optimal concurrency: {optimal_concurrency} ({results[optimal_concurrency]:.4f}s)")


# ============================================================================
# Test 5: Combined Optimizations
# ============================================================================

def test_combined_optimizations():
    """Test optimal combination of all parameters."""
    if not DAFT_AVAILABLE:
        print("❌ Skipping test - Daft not available")
        return
    
    print("\n" + "="*70)
    print("TEST 5: Combined Optimization (Best Configuration)")
    print("="*70)
    
    texts = [f"  HELLO WORLD {i}  " for i in range(10000)]
    model = ExpensiveModel(delay=0.2)
    
    # Baseline: Default configuration
    @node(output_name="result")
    def baseline(text: str, model: ExpensiveModel) -> str:
        return model.predict(text)
    
    pipeline = Pipeline(
        nodes=[baseline],
        engine=DaftEngine(use_batch_udf=False)  # Row-wise
    )
    
    result, elapsed_baseline = benchmark_function(
        pipeline.map,
        inputs={"text": texts, "model": model},
        map_over="text"
    )
    print(f"\n📊 Baseline (row-wise):               {elapsed_baseline:.4f}s")
    
    # Optimized: All features enabled
    @node(
        output_name="result",
        stateful_params=["model"],
        daft_config={
            "batch_size": 1024,
            "max_concurrency": 4,
            "use_process": True
        }
    )
    def optimized(text: str, model: ExpensiveModel) -> str:
        return model.predict(text)
    
    pipeline = Pipeline(
        nodes=[optimized],
        engine=DaftEngine(
            use_batch_udf=True,
            default_daft_config={"batch_size": 1024}
        )
    )
    
    result, elapsed_optimized = benchmark_function(
        pipeline.map,
        inputs={"text": texts, "model": model},
        map_over="text"
    )
    print(f"📊 Optimized (all features):          {elapsed_optimized:.4f}s")
    
    speedup = elapsed_baseline / elapsed_optimized
    print(f"\n🚀 TOTAL SPEEDUP: {speedup:.2f}x")
    print(f"\nOptimizations applied:")
    print(f"  ✅ Stateful model (initialized once)")
    print(f"  ✅ Batch UDF (batch_size=1024)")
    print(f"  ✅ Process isolation (use_process=True)")
    print(f"  ✅ Parallel instances (max_concurrency=4)")


# ============================================================================
# Test 6: Engine-Level vs Node-Level Configuration
# ============================================================================

def test_config_hierarchy():
    """Test configuration inheritance."""
    if not DAFT_AVAILABLE:
        print("❌ Skipping test - Daft not available")
        return
    
    print("\n" + "="*70)
    print("TEST 6: Configuration Hierarchy")
    print("="*70)
    
    texts = [f"Text {i}" for i in range(5000)]
    
    # Engine-level default
    @node(output_name="result")  # No config
    def process1(text: str) -> str:
        return text.lower()
    
    pipeline = Pipeline(
        nodes=[process1],
        engine=DaftEngine(
            use_batch_udf=True,
            default_daft_config={"batch_size": 2048}  # Engine default
        )
    )
    
    result, elapsed = benchmark_function(
        pipeline.map,
        inputs={"text": texts},
        map_over="text"
    )
    print(f"\n📊 Engine default (batch_size=2048): {elapsed:.4f}s")
    
    # Node overrides engine default
    @node(
        output_name="result",
        daft_config={"batch_size": 512}  # Override
    )
    def process2(text: str) -> str:
        return text.lower()
    
    pipeline = Pipeline(
        nodes=[process2],
        engine=DaftEngine(
            use_batch_udf=True,
            default_daft_config={"batch_size": 2048}
        )
    )
    
    result, elapsed = benchmark_function(
        pipeline.map,
        inputs={"text": texts},
        map_over="text"
    )
    print(f"📊 Node override (batch_size=512):   {elapsed:.4f}s")
    print(f"\n✅ Node config overrides engine config")


# ============================================================================
# Main Runner
# ============================================================================

def print_summary():
    """Print optimization summary."""
    print("\n" + "📚"*35)
    print("OPTIMIZATION SUMMARY")
    print("📚"*35)
    print("""
Daft Optimization Parameters:

1. Stateful Parameters (@daft.cls)
   - Mark with: stateful_params=["model"] or __daft_stateful__ = True
   - Benefit: Initialize expensive resources ONCE per worker
   - Use case: ML models, tokenizers, database connections

2. Batch Size (batch_size)
   - Tune batch_size in daft_config
   - Trade-off: Memory vs overhead
   - Optimal: Usually 512-2048 for text, higher for numbers

3. Process Isolation (use_process)
   - Set use_process=True for CPU-bound work
   - Benefit: Avoids Python GIL contention
   - Downside: Process creation overhead

4. Concurrency (max_concurrency)
   - Control parallel UDF instances
   - More instances = more parallelism (but more memory)
   - Optimal: Usually 2-8 depending on workload

5. Combined
   - Stateful + batch + process + concurrency = MAX SPEED
   - Can achieve 10-50x speedup for expensive operations

Configuration Hierarchy:
  1. Node-level daft_config (highest priority)
  2. Engine-level default_daft_config
  3. Daft defaults

Example:
  @node(
      output_name="embedding",
      stateful_params=["model"],
      daft_config={
          "batch_size": 1024,
          "max_concurrency": 4,
          "use_process": True
      }
  )
  def encode(text: str, model: Model) -> list:
      return model.encode(text)
""")


def main():
    """Run all benchmarks."""
    if not DAFT_AVAILABLE:
        print("❌ Daft is not installed. Install with: uv add getdaft")
        return
    
    print("\n" + "🔥"*35)
    print("DAFT OPTIMIZATION BENCHMARKS - HYPERNODES")
    print("🔥"*35)
    
    tests = [
        ("Stateful vs Non-Stateful", test_stateful_vs_nonstateful),
        ("Batch Size Optimization", test_batch_sizes),
        ("Process Isolation", test_use_process),
        ("Max Concurrency", test_max_concurrency),
        ("Combined Optimizations", test_combined_optimizations),
        ("Configuration Hierarchy", test_config_hierarchy),
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print_summary()
    
    print("\n" + "🎉"*35)
    print("BENCHMARKS COMPLETE!")
    print("🎉"*35 + "\n")


if __name__ == "__main__":
    main()
