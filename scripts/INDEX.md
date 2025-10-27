# HyperNodes vs Daft Benchmark Suite - Index

## 📁 Files Overview

| File | Purpose | Audience |
|------|---------|----------|
| **benchmark_hypernodes_vs_daft.py** | Main benchmark script | Developers |
| **BENCHMARK_RESULTS.md** | Detailed analysis and results | Technical readers |
| **README_BENCHMARK.md** | Comprehensive guide | All users |
| **BENCHMARK_SUMMARY.txt** | Quick results summary | Quick reference |
| **QUICK_REFERENCE.md** | Side-by-side comparison | Developers |
| **INDEX.md** | This file | Navigation |

## 🚀 Quick Start

### Run the Benchmark
```bash
cd /Users/giladrubin/python_workspace/hypernodes
uv run python scripts/benchmark_hypernodes_vs_daft.py
```

### View Results
```bash
cat scripts/BENCHMARK_SUMMARY.txt
```

### Read Full Analysis
Open `scripts/BENCHMARK_RESULTS.md` in your editor

## 📊 What's Benchmarked

### 5 Comprehensive Scenarios

1. **Simple Text Processing** (1000 items)
   - Cleaning, tokenizing, counting
   - Tests: Basic transformations
   - Winner: Daft Built-in (13x faster)

2. **Stateful Processing** (500 items)
   - Encoder with expensive initialization
   - Tests: Object reuse and lazy initialization
   - Winner: HyperNodes (3.7x faster)

3. **Batch Vectorized Operations** (1000 items)
   - Numerical normalization
   - Tests: Vectorization capabilities
   - Winner: Daft Built-in (16x faster)

4. **Nested Pipelines** (250 docs, 50 queries)
   - Retrieval-like workflow
   - Tests: Complex multi-stage processing
   - Winner: Daft UDF (35% faster)

5. **Generator Functions** (1000 sentences)
   - One-to-many transformations
   - Tests: Row expansion capabilities
   - Winner: Daft Built-in (2x faster)

## 🎯 Key Findings

### Performance Summary
- **Daft wins**: 4/5 benchmarks
- **HyperNodes wins**: 1/5 benchmarks (stateful processing)
- **Biggest speedup**: Daft built-in batch ops (16x)
- **HyperNodes advantage**: Pre-initialized objects (3.7x)

### When to Use Each

**Use HyperNodes for:**
- Explicit DAG control and visualization
- Fine-grained caching
- Pre-initialized expensive objects
- Modular, reusable components
- Complex branching logic

**Use Daft for:**
- Large-scale data processing (>1GB)
- Vectorized operations
- Automatic optimization
- Distributed execution
- Maximum performance

## 📚 Documentation Structure

### For Quick Answers
1. Start with **QUICK_REFERENCE.md** - Side-by-side code examples
2. Check **BENCHMARK_SUMMARY.txt** - Results at a glance

### For Deep Understanding
1. Read **README_BENCHMARK.md** - Comprehensive guide
2. Study **BENCHMARK_RESULTS.md** - Detailed analysis
3. Run **benchmark_hypernodes_vs_daft.py** - Hands-on experience

### For Implementation
1. Review code in **benchmark_hypernodes_vs_daft.py**
2. Check **QUICK_REFERENCE.md** for patterns
3. Consult **../notebooks/DAFT_TRANSLATION_GUIDE.md** for translation patterns

## 🔧 Configuration

### Scale Factors
Edit `CURRENT_SCALE` in `benchmark_hypernodes_vs_daft.py`:
- `"small"`: 100 items (quick test)
- `"medium"`: 1000 items (default)
- `"large"`: 5000 items (stress test)

### Execution Modes (HyperNodes)
```python
backend = LocalBackend(
    node_execution="sequential",  # or "threaded", "async", "parallel"
    map_execution="sequential",   # or "threaded", "async", "parallel"
    max_workers=8
)
```

## 📈 Results at a Glance

```
Benchmark                      HyperNodes      Daft UDF        Daft Built-in
--------------------------------------------------------------------------------
1. Text Processing             0.1486s         0.6000s         0.0715s ⭐
2. Stateful Processing         0.0299s ⭐       0.1095s         N/A
3. Batch Operations            0.1393s         0.1120s         0.0085s ⭐
4. Nested Pipelines            0.3706s         0.2398s ⭐       N/A
5. Generators                  0.0432s         0.0292s         0.0198s ⭐
```

## 🔗 Related Resources

### In This Repository
- **Translation Guide**: `../notebooks/DAFT_TRANSLATION_GUIDE.md`
- **Daft UDF Guide**: `../guides/daft-udfs.md`
- **Daft PDF Guide**: `../guides/daft-pdf.md`
- **HyperNodes Docs**: `../docs/`

### External
- [Daft Documentation](https://www.getdaft.io/)
- [Daft GitHub](https://github.com/Eventual-Inc/Daft)
- [HyperNodes GitHub](https://github.com/yourusername/hypernodes)

## 🛠️ Extending the Benchmark

### Adding New Scenarios

1. **Define Test Data**
   ```python
   test_data = [...]
   ```

2. **Implement HyperNodes Version**
   ```python
   @node(output_name="result")
   def process_hn(input: Type) -> Type:
       return ...
   ```

3. **Implement Daft UDF Version**
   ```python
   @daft.func
   def process_daft(input: Type) -> Type:
       return ...
   ```

4. **Implement Daft Built-in Version** (if possible)
   ```python
   df = df.with_column("result", df["input"].builtin_op())
   ```

5. **Add Timing and Comparison**
   ```python
   start = time.perf_counter()
   # ... execute
   elapsed = time.perf_counter() - start
   ```

6. **Update Documentation**
   - Add to BENCHMARK_RESULTS.md
   - Update BENCHMARK_SUMMARY.txt
   - Update this INDEX.md

## 📝 File Descriptions

### benchmark_hypernodes_vs_daft.py
**Type**: Python script (560 lines)  
**Purpose**: Main benchmark implementation  
**Contains**:
- 5 comprehensive benchmark scenarios
- HyperNodes implementations
- Daft UDF implementations
- Daft built-in implementations
- Timing and comparison logic
- Results summary

**Key Features**:
- Configurable scale factors
- Multiple execution modes
- Detailed output
- Verification of results

### BENCHMARK_RESULTS.md
**Type**: Markdown documentation  
**Purpose**: Detailed analysis and results  
**Contains**:
- Scenario descriptions
- Performance results
- Key insights
- When to use each framework
- Execution mode comparisons
- Scaling characteristics

**Audience**: Technical readers who want deep understanding

### README_BENCHMARK.md
**Type**: Markdown documentation  
**Purpose**: Comprehensive guide  
**Contains**:
- Overview and quick start
- Detailed scenario descriptions
- Configuration options
- Results summary
- Extension guide
- Related resources

**Audience**: All users, from beginners to advanced

### BENCHMARK_SUMMARY.txt
**Type**: Plain text summary  
**Purpose**: Quick reference  
**Contains**:
- Results table
- Winners by category
- Key insights
- Recommendations
- Quick commands

**Audience**: Users who want quick answers

### QUICK_REFERENCE.md
**Type**: Markdown reference  
**Purpose**: Side-by-side comparison  
**Contains**:
- Feature comparison table
- Code examples
- Decision tree
- Common patterns
- Tips and tricks

**Audience**: Developers implementing solutions

### INDEX.md
**Type**: Markdown index (this file)  
**Purpose**: Navigation and overview  
**Contains**:
- File descriptions
- Quick start guide
- Results summary
- Documentation structure

**Audience**: All users for navigation

## 🎓 Learning Path

### Beginner
1. Read **BENCHMARK_SUMMARY.txt** (5 min)
2. Skim **QUICK_REFERENCE.md** (10 min)
3. Run the benchmark (2 min)

### Intermediate
1. Read **README_BENCHMARK.md** (20 min)
2. Study **QUICK_REFERENCE.md** (15 min)
3. Examine **benchmark_hypernodes_vs_daft.py** (30 min)
4. Run benchmark with different scales (10 min)

### Advanced
1. Read **BENCHMARK_RESULTS.md** (30 min)
2. Study **benchmark_hypernodes_vs_daft.py** in detail (60 min)
3. Modify and extend benchmarks (varies)
4. Contribute improvements

## 💡 Tips

### Running Benchmarks
- Start with "small" scale for quick tests
- Use "medium" for realistic comparisons
- Use "large" to test scaling behavior
- Run multiple times for consistent results

### Interpreting Results
- First run may include JIT compilation overhead
- Focus on relative performance, not absolute times
- Consider your specific use case
- Test with your actual data when possible

### Making Decisions
- Don't optimize prematurely
- Profile your actual workload
- Consider maintainability
- Think about team expertise

## 🤝 Contributing

To improve the benchmark suite:

1. **Add scenarios**: Follow the structure in the main script
2. **Improve analysis**: Update BENCHMARK_RESULTS.md
3. **Enhance docs**: Keep all files in sync
4. **Share findings**: Document interesting discoveries

## 📞 Support

For questions or issues:
- Check the documentation files
- Review the code comments
- Run the benchmarks yourself
- Consult the related resources

## 🏁 Conclusion

This benchmark suite provides a comprehensive comparison between HyperNodes and Daft. Both frameworks have their strengths:

- **Daft**: Performance, automatic optimization, built-in operations
- **HyperNodes**: Explicit control, caching, pre-initialized objects

Choose based on your specific needs, or consider using both in a hybrid approach.

---

**Last Updated**: 2025-10-27  
**Version**: 1.0  
**Maintainer**: HyperNodes Team
