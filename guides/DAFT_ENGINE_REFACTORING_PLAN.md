# DaftEngine SOLID Refactoring Plan

## Executive Summary

The current DaftEngine is 4,032 lines with ~40% (1,600 lines) dedicated to serialization workarounds. This plan refactors it into a clean SOLID architecture with ~1,300 lines, reducing complexity by 70%.

**Key Insight**: Separate core execution logic from distributed serialization concerns. Most serialization complexity may only be needed for Modal/distributed execution, not local Daft execution.

---

## Current State Analysis

### Pain Points
1. **Serialization Complexity** - 1,600 lines, 5 overlapping layers of `__module__` fixing
2. **Map Operations** - 268 lines of complex row ID tracking
3. **Type Inference** - Incomplete Python→Daft type mapping
4. **Stateful Objects** - Multiple overlapping detection heuristics
5. **Code Generation** - Doubles complexity with branching throughout
6. **Debugging** - Lazy evaluation + distributed execution = hard to diagnose

### Current Metrics
- **Total Lines**: 4,032
- **Serialization**: ~1,600 lines (40%)
- **Core Logic**: ~2,400 lines (60%)
- **Methods/Functions**: ~70
- **Cyclomatic Complexity**: Very High

---

## Target Architecture (SOLID Principles)

### Core Classes

```
DaftEngine (Orchestrator)
├── PipelineCompiler (Pipeline → DataFrame)
├── NodeConverter (Node → UDF, uses strategies)
├── StatefulUDFBuilder (@daft.cls wrapper)
├── MapOperationHandler (map_over operations)
├── TypeInferencer (Python → Daft types)
├── OutputMaterializer (DataFrame → Python, uses strategies)
├── DependencyResolver (Topological sort)
└── SerializationManager (OPTIONAL, only if needed)
```

### File Structure

```
src/hypernodes/integrations/daft/
├── engine.py                    (~200 lines - orchestration)
├── compiler.py                  (~300 lines - pipeline compilation)
├── node_converter.py            (~200 lines - node → UDF)
├── stateful_udf.py             (~150 lines - @daft.cls handling)
├── map_operations.py           (~150 lines - map_over)
├── type_inference.py           (~100 lines - type mapping)
├── output_materializer.py      (~100 lines - output conversion)
├── dependency_resolver.py      (~50 lines - topological sort)
└── serialization.py            (~200-300 lines - ONLY if needed)

Total: ~1,250-1,350 lines (70% reduction!)
```

### SOLID Principles Application

- **Single Responsibility**: Each class has one reason to change
- **Open/Closed**: Strategy pattern for extensibility (node converters, materializers)
- **Liskov Substitution**: Interchangeable strategies
- **Interface Segregation**: Small, focused interfaces
- **Dependency Inversion**: Depend on abstractions (Engine interface)

---

## Implementation Phases

### Phase 0: Validate Serialization Necessity (2-3 hours) ⭐ START HERE

**Goal**: Prove whether 1,600 lines of serialization code are actually needed.

#### Steps

1. **Create `engine_minimal.py`**
   - Clean SOLID implementation
   - NO serialization workarounds
   - NO `_fix_*` methods
   - ~500-600 lines total

2. **Run Test Suite**
   ```bash
   uv run pytest tests/test_daft_backend.py -v
   uv run pytest tests/test_daft_backend_complex_types.py -v
   uv run pytest tests/test_daft_backend_map_over.py -v
   uv run python scripts/test_exact_repro.py
   ```

3. **Document Failures**
   - Create `serialization_requirements.md`
   - What breaks? Local or only Modal?
   - Can simpler fixes work?

4. **Add Minimal Serialization**
   - Only where tests fail
   - Track lines added vs benefit

#### Expected Outcomes

- **Best Case**: Most tests pass! 80% reduction possible.
- **Likely Case**: Some failures, simpler fixes work. 75% reduction.
- **Worst Case**: Need full serialization, but validate necessity. 50% reduction via consolidation.

---

### Phase 1: Implement SOLID Components (4-6 hours)

#### 1.1 Create `type_inference.py`
```python
class TypeInferencer:
    """Maps Python type hints to Daft data types."""
    TYPE_MAP = {...}

    def infer(self, func: Callable) -> daft.DataType
    def _map_type(self, python_type) -> daft.DataType
```

**Tests**: Unit tests for type mapping

#### 1.2 Create `dependency_resolver.py`
```python
class DependencyResolver:
    """Topological sort of pipeline nodes."""

    def sort(self, nodes: List) -> List
    def _build_graph(self, nodes: List) -> nx.DiGraph
```

**Tests**: Unit tests for dependency resolution

#### 1.3 Create `stateful_udf.py`
```python
class StatefulUDFBuilder:
    """Builds @daft.cls wrappers for stateful objects."""

    def build(self, func, stateful_params, dynamic_params)
    def _extract_daft_cls_config(self, stateful_params) -> Dict
```

**Tests**: Unit tests for stateful UDF building

#### 1.4 Create `output_materializer.py`
```python
class OutputMaterializer:
    """Converts Daft DataFrame to Python objects."""
    # Strategy pattern for different output formats

    def materialize(self, df: daft.DataFrame, output_name) -> Dict
    def _via_pydict(self, df) -> Dict
    def _via_arrow(self, df) -> Dict
    def _via_pandas(self, df) -> Dict
```

**Tests**: Unit tests for each strategy

#### 1.5 Create `map_operations.py`
```python
class MapOperationHandler:
    """Handles map_over using explode/groupby."""

    def handle(self, node, df, stateful_inputs) -> daft.DataFrame
    def _apply_mapping(self, df, mapping) -> daft.DataFrame
```

**Tests**: Unit tests for map operations

#### 1.6 Create `node_converter.py`
```python
class NodeConverter:
    """Converts nodes to Daft UDFs."""
    # Strategy pattern for different node types

    def convert(self, node, df, stateful_inputs) -> daft.DataFrame
    def _convert_function_node(self, node, df, stateful_inputs)
    def _convert_pipeline_node(self, node, df, stateful_inputs)
    def _convert_mapped_pipeline_node(self, node, df, stateful_inputs)
```

**Tests**: Unit tests for each converter strategy

#### 1.7 Create `compiler.py`
```python
class PipelineCompiler:
    """Compiles Pipeline to Daft DataFrame operations."""

    def compile(self, pipeline, inputs) -> daft.DataFrame
    def compile_map(self, pipeline, items, inputs) -> daft.DataFrame
    def _split_inputs(self, inputs) -> Tuple[Dict, Dict]
```

**Tests**: Integration tests with real pipelines

#### 1.8 Refactor `engine.py`
```python
class DaftEngine(Engine):
    """Orchestrates pipeline execution using composed objects."""

    def __init__(self, collect=True, debug=False, generate_code=False)
    def run(self, pipeline, inputs, output_name) -> Dict
    def map(self, pipeline, items, inputs, output_name) -> List[Dict]
```

**Tests**: Full integration tests

---

### Phase 2: Incremental Testing (2-3 hours)

#### Testing Strategy
- Test after each component creation
- Maintain green test suite throughout
- Fix issues immediately

#### Test Commands
```bash
# After each component:
uv run pytest tests/test_daft_backend.py::test_name -v

# Full Daft test suite:
uv run pytest tests/test_daft*.py -v

# Complex example:
uv run python scripts/test_exact_repro.py

# Performance check:
uv run python scripts/benchmark_daft_vs_native.py
```

---

### Phase 3: Code Generation Support (2-3 hours)

#### Goal
Support optional code generation alongside execution (as flag).

#### Design
```python
class DaftEngine(Engine):
    def __init__(
        self,
        collect: bool = True,
        debug: bool = False,
        generate_code: bool = False,  # NEW FLAG
        output_code_path: str = None,
    ):
        self.generate_code = generate_code
        self.output_code_path = output_code_path

        if generate_code:
            self._code_generator = DaftCodeGenerator()
```

#### Implementation
```python
class DaftCodeGenerator:
    """Generates standalone Daft code."""

    def __init__(self):
        self.imports = set()
        self.code_lines = []

    def record_operation(self, operation_type, **kwargs):
        """Record a DataFrame operation."""
        # Called by converter/compiler during execution
        pass

    def generate(self) -> str:
        """Generate complete code."""
        return self._format_code()
```

**Integration**: Each component optionally calls `code_generator.record_operation()` when flag is set.

---

### Phase 4: Documentation & Cleanup (1-2 hours)

#### Documentation
1. **Architecture diagram** - Component relationships with Graphviz
2. **API guide** - How to use new structure
3. **Migration guide** - Changes from old implementation
4. **Troubleshooting guide** - Common issues and fixes

#### Code Cleanup
1. Rename old implementation: `engine.py` → `engine_legacy.py`
2. Update imports throughout codebase
3. Add comprehensive docstrings
4. Add type hints everywhere
5. Run linters: `ruff`, `mypy`

---

## Success Metrics

1. ✅ **Simplicity**: 70%+ reduction in lines (4,032 → ~1,300)
2. ✅ **Maintainability**: Each class < 300 lines, single responsibility
3. ✅ **Testability**: Each component testable in isolation
4. ✅ **Functionality**: All existing tests pass
5. ✅ **Performance**: No regressions vs current
6. ✅ **Clarity**: New developer understands in < 30 min
7. ✅ **Code Generation**: Optional code generation works

---

## Risk Mitigation

1. **Feature branch**: `feature/daft-engine-solid-refactor`
2. **Keep old implementation**: Rename to `engine_legacy.py`
3. **Incremental approach**: Test after each component
4. **Validation first**: Prove serialization necessity
5. **Backwards compatibility**: Same public API

---

## Timeline

- **Phase 0** (validation): 2-3 hours
- **Phase 1** (components): 4-6 hours
- **Phase 2** (testing): 2-3 hours
- **Phase 3** (code gen): 2-3 hours
- **Phase 4** (docs): 1-2 hours

**Total: 11-17 hours** (1.5-2 work days)

---

## Design Highlights

### 1. Composition Over Inheritance
```python
class DaftEngine:
    def __init__(self):
        self._compiler = PipelineCompiler()
        self._materializer = OutputMaterializer()
        # Clear dependencies, easy to test
```

### 2. Strategy Pattern
- Different node converters for different node types
- Different materializers for different output formats
- Easy to extend without modifying existing code

### 3. Optional Features
- Serialization only when needed (distributed mode)
- Code generation only when flag set
- Separate concerns, cleaner core

### 4. Clear Data Flow
```
inputs → split → DataFrame → compile → collect → materialize → outputs
              ↓                    ↓
         stateful          per-node conversion
                                   ↓
                            UDF application
```

---

## Questions Resolved

1. **Serialization necessity**: Validate in Phase 0
2. **Breaking changes**: Avoid by keeping same public API
3. **Performance**: Test after each phase
4. **Priority**: Core logic first, optimizations later
5. **Code generation**: Support as optional flag

---

## Next Steps

1. ✅ Create this plan document
2. ⏭️ Create feature branch
3. ⏭️ Implement Phase 0: Minimal engine
4. ⏭️ Run tests and document findings
5. ⏭️ Proceed with Phase 1 based on findings
