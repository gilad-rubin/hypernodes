# Daft Engine Code Generation Migration Plan

## Overview

This document outlines the plan to:
1. **Update tests** to use the new `output_mode` parameter (accepts "dict" or "daft")
2. **Port code generation** from legacy engine to the new modular architecture
3. **Remove legacy code** and backwards compatibility layer

## Current State

### New Engine Architecture (Modular)
```
DaftEngine
├── StatefulUDFBuilder - builds stateful UDFs
├── NodeConverter - converts nodes to Daft UDFs
├── PipelineCompiler - compiles pipeline to Daft operations
├── MapOperationHandler - handles map operations
└── OutputMaterializer - materializes results (dict or daft)
```

### Legacy Engine (Monolithic)
- Single large class with intertwined execution and code generation
- Code generation tracks operations as they execute
- Generates complete executable Python code with UDF definitions

## Task 1: Update Tests for `output_mode`

### Changes Required

**File: `tests/test_daft_return_formats.py`**

Replace `python_return_strategy` with `output_mode`:
- Remove: `python_return_strategy="pydict"` (legacy behavior)
- Remove: `python_return_strategy="pandas"` (legacy behavior)  
- Remove: `python_return_strategy="invalid"` (error test)
- Add: `output_mode="dict"` (new default - returns Python dicts)
- Add: `output_mode="daft"` (returns raw Daft DataFrame)
- Add: `output_mode="invalid"` (error test)

### Test Structure After Changes

```python
def test_daft_engine_output_mode_dict():
    """Test that output_mode='dict' returns Python dicts."""
    pipeline = _build_simple_pipeline().with_engine(DaftEngine(output_mode="dict"))
    result = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    assert isinstance(result, dict)
    assert result["doubled"] == [2, 4, 6]

def test_daft_engine_output_mode_daft():
    """Test that output_mode='daft' returns Daft DataFrame."""
    pipeline = _build_simple_pipeline().with_engine(DaftEngine(output_mode="daft"))
    result = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
    assert hasattr(result, "to_pydict")
    assert result.to_pydict()["doubled"] == [2, 4, 6]

def test_daft_engine_invalid_output_mode():
    """Test that invalid output_mode raises ValueError."""
    with pytest.raises(ValueError, match="Invalid output mode"):
        DaftEngine(output_mode="invalid")
```

## Task 2: Port Code Generation to New Engine

### Components to Port

#### 1. Code Generation Infrastructure

**New file: `src/hypernodes/integrations/daft/code_generator.py`**

```python
class CodeGenerator:
    """Generates executable Daft code from pipeline compilation results."""
    
    def __init__(self):
        self._imports: Set[Tuple[str, str]] = set()
        self._udf_definitions: List[str] = []
        self._operation_lines: List[str] = []
        self._stateful_inputs: Dict[str, Any] = {}
    
    def generate_code(
        self,
        compilation: Compilation,
        inputs: Dict[str, Any],
    ) -> str:
        """Generate complete executable Python code."""
        ...
```

Key methods to implement:
- `generate_code()` - main entry point
- `_generate_header()` - docstring with performance analysis
- `_generate_imports()` - import statements
- `_generate_stateful_setup()` - stateful object initialization code
- `_generate_udf_definitions()` - UDF function definitions
- `_generate_pipeline_operations()` - DataFrame transformations
- `_format_value_for_code()` - convert Python values to code strings

#### 2. Integration with Existing Components

**Modify: `src/hypernodes/integrations/daft/node_converter.py`**
- Add `generate_udf_code()` method to produce UDF definition as string
- Track imports needed for each UDF

**Modify: `src/hypernodes/integrations/daft/compiler.py`**
- Add optional code generation tracking
- Store operation sequence for code generation
- Return metadata about operations performed

**Modify: `src/hypernodes/integrations/daft/stateful_udf.py`**
- Add `generate_code()` method for stateful UDF definitions
- Track which inputs are stateful objects

#### 3. Engine Integration

**Modify: `src/hypernodes/integrations/daft/engine.py`**

Add code generation mode that works with new architecture:

```python
class DaftEngine(Engine):
    def __init__(
        self,
        collect: bool = True,
        output_mode: str = "dict",
        *,
        debug: bool = False,
        code_generation_mode: bool = False,
    ):
        # Remove legacy delegation
        # Initialize code generator if in code_generation_mode
        if code_generation_mode:
            self._code_generator = CodeGenerator()
        
        # Initialize normal components
        self._stateful_builder = StatefulUDFBuilder()
        self._node_converter = NodeConverter(
            self._stateful_builder,
            code_generator=self._code_generator if code_generation_mode else None
        )
        ...
```

Add methods:
```python
def get_generated_code(self) -> str:
    """Return complete executable Daft code."""
    if not self.code_generation_mode:
        return "Code generation mode not enabled."
    return self._code_generator.generate_code(...)

@property
def generated_code(self) -> str:
    """Compatibility property for Pipeline.show_daft_code()."""
    ...
```

### Implementation Strategy

**Phase 1: Create Code Generator Infrastructure**
1. Create `code_generator.py` with `CodeGenerator` class
2. Implement basic code generation (header, imports, structure)
3. Add tests for code generator in isolation

**Phase 2: Integrate with Components**
1. Add code tracking to `NodeConverter`
2. Add code tracking to `PipelineCompiler`
3. Add code tracking to `MapOperationHandler`
4. Ensure generated code matches what the components actually do

**Phase 3: Engine Integration**
1. Remove legacy engine delegation in `__init__`
2. Wire up code generator to all components
3. Implement `get_generated_code()` and properties
4. Add compilation result tracking

**Phase 4: Testing & Validation**
1. Run existing code generation tests
2. Compare generated code between legacy and new
3. Verify generated code executes correctly
4. Test edge cases (nested pipelines, map operations, stateful UDFs)

**Phase 5: Cleanup**
1. Remove `code_generation_mode` delegation to legacy
2. Remove legacy engine runtime fallbacks
3. Consider deleting `engine_legacy.py` (or mark deprecated)
4. Update documentation

## Files to Modify

### Create New Files
- `src/hypernodes/integrations/daft/code_generator.py` - code generation logic

### Modify Existing Files
- `src/hypernodes/integrations/daft/engine.py` - integrate code generator
- `src/hypernodes/integrations/daft/node_converter.py` - add code tracking
- `src/hypernodes/integrations/daft/compiler.py` - add code tracking
- `src/hypernodes/integrations/daft/stateful_udf.py` - add code generation
- `src/hypernodes/integrations/daft/map_operations.py` - add code tracking
- `tests/test_daft_return_formats.py` - update for output_mode

### Eventually Delete/Deprecate
- `src/hypernodes/integrations/daft/engine_legacy.py` - after migration complete

## Tests to Update

### Immediate (Task 1)
- `tests/test_daft_return_formats.py` - replace python_return_strategy with output_mode

### After Migration (Task 2)
- `tests/test_daft_code_generation.py` - verify with new engine
- `tests/test_daft_code_execution_equivalence.py` - verify equivalence
- `tests/test_daft_ellipsis_fix.py` - verify no ellipsis in generated code

## Validation Criteria

### Task 1 Complete When:
- [x] All tests use `output_mode` instead of `python_return_strategy`
- [x] Tests pass with `output_mode="dict"` and `output_mode="daft"`
- [x] Error handling works for invalid output modes
- [x] No breaking changes to existing functionality

### Task 2 Complete When:
- [ ] Code generation works without legacy engine delegation
- [ ] Generated code is executable and produces correct results
- [ ] All existing code generation tests pass
- [ ] Generated code matches quality of legacy implementation
- [ ] Performance analysis included in generated code
- [ ] Stateful UDFs handled correctly
- [ ] Map operations generate correct explode/groupby code
- [ ] Nested pipelines supported
- [ ] Legacy engine can be deleted without breaking functionality

## Risk Assessment

### Low Risk (Task 1)
- Simple parameter rename with validation
- Clear mapping: pydict/pandas/arrow → dict, daft → daft
- All functionality preserved

### Medium Risk (Task 2)
- Code generation is complex
- Must maintain exact equivalence with runtime behavior
- Integration with multiple components
- Need thorough testing

### Mitigation Strategies
1. **Incremental approach**: Build code generator in isolation first
2. **Parallel validation**: Keep legacy working until new is proven
3. **Comprehensive tests**: Validate generated code executes correctly
4. **Code comparison**: Compare legacy vs new generated code
5. **Gradual deprecation**: Mark legacy as deprecated before deletion

## Timeline Estimate

### Task 1: 1-2 hours
- Update test file
- Run tests to verify
- Simple and low-risk

### Task 2: 8-16 hours
- Phase 1: 2-3 hours (infrastructure)
- Phase 2: 3-5 hours (component integration)
- Phase 3: 2-3 hours (engine integration)
- Phase 4: 2-3 hours (testing & validation)
- Phase 5: 1-2 hours (cleanup)

**Total: 9-18 hours** depending on complexity and edge cases discovered

## Next Steps

1. **Start with Task 1** - quick win, unblocks testing
2. **Design code generator API** - define interfaces before implementation
3. **Build incrementally** - test each phase thoroughly
4. **Maintain backwards compatibility** during migration
5. **Delete legacy code** only after full validation

## Questions to Resolve

1. Should we keep `python_return_strategy` as deprecated alias for backwards compatibility?
   - **Recommendation**: No - clean break since this is a refactoring branch
   
2. Should we delete `engine_legacy.py` immediately or mark deprecated?
   - **Recommendation**: Mark deprecated in Task 2, delete in future PR
   
3. Do we need to support code generation for all map operation types?
   - **Recommendation**: Yes - maintain feature parity with legacy

4. Should generated code include performance warnings for nested maps?
   - **Recommendation**: Yes - very valuable for users

## Success Metrics

- All tests pass without legacy engine
- Generated code executes correctly
- Code quality matches or exceeds legacy
- No functionality regression
- Cleaner, more maintainable architecture
