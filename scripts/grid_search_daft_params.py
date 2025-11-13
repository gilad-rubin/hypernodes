"""
Grid search for optimal Daft hyperparameters.

Tests all combinations of:
- use_process: [True, False]
- max_concurrency: [1, 2, 4, 8]
- batch_size: [128, 512, 1024, 2048, 4096, 8192]

Using stateful objects to maximize performance.
"""

import time
from typing import Any
from itertools import product
import json

try:
    import daft
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False
    print("❌ Daft not available. Install with: uv add getdaft")

from hypernodes import Pipeline, node
from hypernodes.integrations.daft import DaftEngine


class StatefulModel:
    """Expensive stateful model for testing."""
    __daft_stateful__ = True
    
    def __init__(self, delay: float = 0.05):
        time.sleep(delay)  # Simulate loading
        self.counter = 0
    
    def predict(self, text: str) -> str:
        self.counter += 1
        return text.strip().lower()


def benchmark_configuration(
    use_process: bool,
    max_concurrency: int,
    batch_size: int,
    num_items: int = 10000
) -> dict[str, Any]:
    """Benchmark a single configuration."""
    
    # Create stateful model
    model = StatefulModel(delay=0.05)
    
    # Create node
    @node(output_name="result")
    def process(text: str, model: StatefulModel) -> str:
        return model.predict(text)
    
    # Create engine with config
    engine = DaftEngine(
        use_batch_udf=True,
        default_daft_config={
            "use_process": use_process,
            "max_concurrency": max_concurrency,
            "batch_size": batch_size
        }
    )
    
    pipeline = Pipeline(nodes=[process], engine=engine)
    
    # Create test data
    texts = [f"  TEXT {i}  " for i in range(num_items)]
    
    # Benchmark
    try:
        start = time.time()
        result = pipeline.map(
            inputs={"text": texts, "model": model},
            map_over="text"
        )
        elapsed = time.time() - start
        
        # Verify correctness
        assert len(result) == num_items, f"Expected {num_items} results, got {len(result)}"
        
        return {
            "use_process": use_process,
            "max_concurrency": max_concurrency,
            "batch_size": batch_size,
            "time": elapsed,
            "throughput": num_items / elapsed,
            "status": "success"
        }
    except Exception as e:
        return {
            "use_process": use_process,
            "max_concurrency": max_concurrency,
            "batch_size": batch_size,
            "time": float('inf'),
            "throughput": 0,
            "status": f"error: {str(e)[:50]}"
        }


def run_grid_search():
    """Run comprehensive grid search."""
    if not DAFT_AVAILABLE:
        print("❌ Daft not installed")
        return
    
    print("\n" + "🔍"*35)
    print("DAFT HYPERPARAMETER GRID SEARCH")
    print("🔍"*35)
    
    # Define search space
    use_process_values = [False, True]
    max_concurrency_values = [1, 2, 4, 8]
    batch_size_values = [128, 512, 1024, 2048, 4096, 8192]
    
    total_configs = len(use_process_values) * len(max_concurrency_values) * len(batch_size_values)
    
    print(f"\n📊 Testing {total_configs} configurations...")
    print(f"   use_process: {use_process_values}")
    print(f"   max_concurrency: {max_concurrency_values}")
    print(f"   batch_size: {batch_size_values}")
    print(f"   items: 10,000")
    print()
    
    # Run grid search
    results = []
    
    for i, (use_process, max_concurrency, batch_size) in enumerate(
        product(use_process_values, max_concurrency_values, batch_size_values), 1
    ):
        print(f"[{i}/{total_configs}] Testing: "
              f"use_process={use_process}, "
              f"max_concurrency={max_concurrency}, "
              f"batch_size={batch_size}...", 
              end=" ", flush=True)
        
        result = benchmark_configuration(
            use_process=use_process,
            max_concurrency=max_concurrency,
            batch_size=batch_size
        )
        
        results.append(result)
        
        if result["status"] == "success":
            print(f"✅ {result['time']:.3f}s ({result['throughput']:.0f} items/s)")
        else:
            print(f"❌ {result['status']}")
    
    return results


def analyze_results(results: list[dict]) -> None:
    """Analyze and display results."""
    print("\n" + "📈"*35)
    print("ANALYSIS")
    print("📈"*35)
    
    # Filter successful results
    successful = [r for r in results if r["status"] == "success"]
    
    if not successful:
        print("❌ No successful runs!")
        return
    
    # Find best configuration
    best = min(successful, key=lambda x: x["time"])
    
    print("\n🏆 BEST CONFIGURATION:")
    print(f"   use_process:      {best['use_process']}")
    print(f"   max_concurrency:  {best['max_concurrency']}")
    print(f"   batch_size:       {best['batch_size']}")
    print(f"   time:             {best['time']:.4f}s")
    print(f"   throughput:       {best['throughput']:.0f} items/s")
    
    # Worst configuration
    worst = max(successful, key=lambda x: x["time"])
    
    print("\n🐌 WORST CONFIGURATION:")
    print(f"   use_process:      {worst['use_process']}")
    print(f"   max_concurrency:  {worst['max_concurrency']}")
    print(f"   batch_size:       {worst['batch_size']}")
    print(f"   time:             {worst['time']:.4f}s")
    print(f"   throughput:       {worst['throughput']:.0f} items/s")
    
    speedup = worst["time"] / best["time"]
    print(f"\n⚡ SPEEDUP: {speedup:.2f}x (best vs worst)")
    
    # Effect of each parameter
    print("\n" + "="*70)
    print("PARAMETER EFFECTS (Average Time)")
    print("="*70)
    
    # Batch size effect
    print("\n📦 Batch Size Effect:")
    batch_groups = {}
    for r in successful:
        batch_size = r["batch_size"]
        if batch_size not in batch_groups:
            batch_groups[batch_size] = []
        batch_groups[batch_size].append(r["time"])
    
    batch_avg = {k: sum(v)/len(v) for k, v in batch_groups.items()}
    batch_sorted = sorted(batch_avg.items(), key=lambda x: x[1])
    worst_batch_time = max(batch_avg.values())
    
    for batch_size, avg_time in batch_sorted:
        speedup_vs_worst = worst_batch_time / avg_time
        print(f"   {batch_size:>5}: {avg_time:.4f}s (speedup: {speedup_vs_worst:.2f}x)")
    
    # Concurrency effect
    print("\n🔄 Max Concurrency Effect:")
    conc_groups = {}
    for r in successful:
        concurrency = r["max_concurrency"]
        if concurrency not in conc_groups:
            conc_groups[concurrency] = []
        conc_groups[concurrency].append(r["time"])
    
    conc_avg = {k: sum(v)/len(v) for k, v in conc_groups.items()}
    conc_sorted = sorted(conc_avg.items(), key=lambda x: x[1])
    worst_conc_time = max(conc_avg.values())
    
    for concurrency, avg_time in conc_sorted:
        speedup_vs_worst = worst_conc_time / avg_time
        print(f"   {concurrency:>2}: {avg_time:.4f}s (speedup: {speedup_vs_worst:.2f}x)")
    
    # Process effect
    print("\n🔓 use_process Effect:")
    proc_groups = {True: [], False: []}
    for r in successful:
        proc_groups[r["use_process"]].append(r["time"])
    
    proc_avg = {k: sum(v)/len(v) for k, v in proc_groups.items() if v}
    proc_sorted = sorted(proc_avg.items(), key=lambda x: x[1])
    worst_proc_time = max(proc_avg.values())
    
    for use_proc, avg_time in proc_sorted:
        speedup_vs_worst = worst_proc_time / avg_time
        label = "True " if use_proc else "False"
        print(f"   {label}: {avg_time:.4f}s (speedup: {speedup_vs_worst:.2f}x)")
    
    # Top 10 configurations
    print("\n" + "="*70)
    print("TOP 10 CONFIGURATIONS")
    print("="*70)
    print()
    
    top10 = sorted(successful, key=lambda x: x["time"])[:10]
    
    print(f"{'Rank':<6} {'Batch':<8} {'Conc':<6} {'Process':<9} {'Time (s)':<10} {'Throughput'}")
    print("-" * 70)
    
    for rank, row in enumerate(top10, 1):
        print(f"{rank:<6} "
              f"{row['batch_size']:<8} "
              f"{row['max_concurrency']:<6} "
              f"{str(row['use_process']):<9} "
              f"{row['time']:<10.4f} "
              f"{row['throughput']:.0f} items/s")
    
    # Interaction effects
    print("\n" + "="*70)
    print("INTERACTION: Batch Size × Max Concurrency (Average Time)")
    print("="*70)
    
    # Group by batch_size and max_concurrency
    interaction = {}
    for r in successful:
        key = (r["batch_size"], r["max_concurrency"])
        if key not in interaction:
            interaction[key] = []
        interaction[key].append(r["time"])
    
    interaction_avg = {k: sum(v)/len(v) for k, v in interaction.items()}
    
    # Get unique values
    batch_sizes = sorted(set(r["batch_size"] for r in successful))
    concurrencies = sorted(set(r["max_concurrency"] for r in successful))
    
    # Print table header
    print(f"\n{'Batch Size':<12}", end="")
    for conc in concurrencies:
        print(f"{conc:>10}", end="")
    print()
    print("-" * (12 + 10 * len(concurrencies)))
    
    # Print rows
    for batch in batch_sizes:
        print(f"{batch:<12}", end="")
        for conc in concurrencies:
            if (batch, conc) in interaction_avg:
                print(f"{interaction_avg[(batch, conc)]:>10.4f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()
    
    # Save results
    output_file = "daft_grid_search_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")


def print_recommendations(results: list[dict]) -> None:
    """Print optimization recommendations."""
    print("\n" + "💡"*35)
    print("RECOMMENDATIONS")
    print("💡"*35)
    
    successful = [r for r in results if r["status"] == "success"]
    
    if not successful:
        return
    
    best = min(successful, key=lambda x: x["time"])
    
    print("\n✅ For optimal performance, configure your engine:")
    print()
    print("```python")
    print("engine = DaftEngine(")
    print("    use_batch_udf=True,")
    print("    default_daft_config={")
    print(f'        "batch_size": {best["batch_size"]},')
    print(f'        "max_concurrency": {best["max_concurrency"]},')
    print(f'        "use_process": {best["use_process"]},')
    print("    }")
    print(")")
    print("```")
    
    print("\n📋 General Guidelines:")
    
    # Batch size
    batch_groups = {}
    for r in successful:
        batch_size = r["batch_size"]
        if batch_size not in batch_groups:
            batch_groups[batch_size] = []
        batch_groups[batch_size].append(r["time"])
    batch_avg = {k: sum(v)/len(v) for k, v in batch_groups.items()}
    best_batch = min(batch_avg, key=batch_avg.get)
    
    print(f"\n1. Batch Size: Use {best_batch}")
    print(f"   - Larger batches reduce overhead")
    print(f"   - Sweet spot: {best_batch}")
    
    # Concurrency
    conc_groups = {}
    for r in successful:
        concurrency = r["max_concurrency"]
        if concurrency not in conc_groups:
            conc_groups[concurrency] = []
        conc_groups[concurrency].append(r["time"])
    conc_avg = {k: sum(v)/len(v) for k, v in conc_groups.items()}
    best_conc = min(conc_avg, key=conc_avg.get)
    
    print(f"\n2. Max Concurrency: Use {best_conc}")
    print(f"   - Too low: underutilizes CPUs")
    print(f"   - Too high: overhead dominates")
    print(f"   - Optimal: {best_conc}")
    
    # Process
    proc_groups = {True: [], False: []}
    for r in successful:
        proc_groups[r["use_process"]].append(r["time"])
    proc_avg = {k: sum(v)/len(v) for k, v in proc_groups.items() if v}
    best_proc = min(proc_avg, key=proc_avg.get)
    
    print(f"\n3. use_process: Use {best_proc}")
    if best_proc:
        print(f"   - Benefit: Avoids Python GIL")
        print(f"   - Cost: Process creation overhead")
        print(f"   - Worth it for this workload ✅")
    else:
        print(f"   - Process overhead not worth it for this workload")


def main():
    """Run grid search and analysis."""
    if not DAFT_AVAILABLE:
        return
    
    # Run grid search
    results = run_grid_search()
    
    if not results:
        print("❌ No results!")
        return
    
    # Analyze results
    analyze_results(results)
    
    # Print recommendations
    print_recommendations(results)
    
    print("\n" + "🎉"*35)
    print("GRID SEARCH COMPLETE!")
    print("🎉"*35 + "\n")


if __name__ == "__main__":
    main()

