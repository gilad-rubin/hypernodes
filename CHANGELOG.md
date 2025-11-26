# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.6] - 2025-11-26

### Added
- **Dark Theme Preview Generation**: Added script and functionality to generate theme previews, especially for dark mode.
- **Adaptive SVG Processing**: Enhanced Graphviz rendering to better handle SVG processing.
- **Theme Improvements**: Improved color management and visualization options for better clarity.

### Refactored
- **Graphviz Theme**: Renamed `GraphvizStyle` to `GraphvizTheme` and updated node styling.
- **Visualization Styles**: Updated theme detection and style configurations for better consistency.

## [0.4.5] - 2025-11-21

### Changed
- **Visualization module reorganization**
  - Created dedicated `src/hypernodes/viz/` package for all visualization code
  - Moved visualization files:
    - `visualization.py` → `viz/visualization.py` (Graphviz rendering and legacy functions)
    - `graph_serializer.py` → `viz/graph_serializer.py` (frontend-agnostic graph serialization)
    - `visualization_engines.py` → `viz/visualization_engines.py` (pluggable rendering engines)
    - `visualization_widget.py` → `viz/visualization_widget.py` (IPyWidget React Flow components)
  - Created `viz/__init__.py` that exports public API for backwards compatibility
  - Updated all imports throughout codebase (core, tests, scripts)
  - Cleaner separation of concerns: serialization, rendering engines, and widgets
  
## [0.4.4] - 2025-11-20

### Fixed
- **DaftEngine: Bound inputs support in nested pipelines**
  - Fixed critical bug where DaftEngine failed to execute nested pipelines with bound inputs
  - `SimplePipelineOperation` and `PipelineNodeOperation` now properly propagate bound inputs to execution context
  - Bound inputs are added to `stateful_inputs` when executing inner nodes, then restored after execution
  - This enables patterns like binding expensive resources (vector stores, LLMs) to inner pipelines
  - Added comprehensive tests in `tests/test_daft_bound_inputs.py`
  - Issue: `ValueError: Parameter 'X' not found for node 'Y'` when using DaftEngine with bound nested pipelines
  
- **DaftEngine: Multi-output node handling in nested pipelines**
  - Fixed handling of tuple output names when updating available columns
  - Both `SimplePipelineOperation` and `PipelineNodeOperation` now properly handle nodes with multiple outputs
  - Fixed aggregation logic in `PipelineNodeOperation` to flatten tuple outputs before creating Daft expressions
  - Prevents `TypeError: 'tuple' object cannot be cast as 'str'` in nested map operations

## [0.4.3] - 2025-11-20

### Fixed
- **PipelineNode output mapping optimization**
  - Fixed selective output execution for nested pipelines with `output_mapping`
  - Inner pipelines now only compute outputs that are explicitly exposed via `output_mapping`
  - This avoids unnecessary computation for unmapped outputs
  - Added comprehensive tests in `tests/test_output_mapping.py`

- **Visualization: Input mapping edge connections**
  - Fixed issue where mapped parameters appeared as floating nodes without connections
  - Edges now correctly connect producer nodes to inner consumers through input mapping
  - Example: `extract_query` → `query` now properly connects to nested pipeline expecting different param name
  - Applied reverse input mapping to show outer parameter names in outer scope
  - Bound parameters from nested pipelines are now correctly displayed with outer names

### Added
- **Pipeline input binding with `.bind()` and `.unbind()`**
  - New `.bind(**inputs)` method to set default input values for pipelines
  - Bound inputs are used as defaults in `.run()` and `.map()` calls
  - Can be overridden by passing `inputs=` at runtime
  - Multiple `.bind()` calls merge inputs (later calls override earlier values for same keys)
  - `.unbind(*keys)` removes specific bound inputs, or all if no keys specified
  - Similar to `functools.partial` - enables partial application of pipeline parameters
  - Common use cases: bind expensive resources (models), set default hyperparameters, simplify API
  - Made `inputs` parameter optional in `.run()` and `.map()` when using `.bind()`
  - Added comprehensive tests in `tests/test_bind.py`

- **Input fulfillment tracking properties**
  - New `.bound_inputs` property on Pipeline and PipelineNode - returns dict of bound values
  - New `.unfulfilled_args` property on Pipeline and PipelineNode - returns tuple of unbound parameter names
  - Validation now respects bound inputs - only unfulfilled parameters are required at runtime
  - Nested pipelines with fully bound inputs no longer require outer pipeline to provide them
  - Enables better introspection: check what's bound vs. what's still needed
  - `Pipeline.__repr__()` now shows bound inputs and unfulfilled args for better debugging
  - `PipelineNode.__repr__()` also shows bound status and unfulfilled requirements
  - **Visualization now shows bound inputs with lighter/transparent color** to distinguish from unfulfilled inputs

- **Visualization: Input/Output mapping indicators**
  - Edge labels now show parameter mappings when using `input_mapping` or `output_mapping`
  - Format: `outer_name → inner_name` on edges crossing nested pipeline boundaries
  - Example: `eval_pairs → eval_pair` shows outer parameter being mapped to inner name
  - Helps understand how parameters flow through nested pipelines with renamed parameters
  - Legend updated with explanation: "a → b: Parameter Mapping"
  - Makes complex pipeline compositions much easier to understand and debug

## [0.4.2] - 2025-11-20

### Fixed
- **PipelineNode input broadcasting bug**
  - Fixed an issue where shared/constant inputs were dropped when using `as_node(map_over=...)` without explicit `input_mapping`.
  - This ensures correct behavior for nested pipelines that map over one input while preserving others as shared context.
  - Verified with regression tests in `tests/test_nested_pipelines.py`.

## [0.4.1] - 2025-11-19

### Changed
- **DualNode batch functions now enforce strict PyArrow input contract**
  - Batch functions MUST accept `pyarrow.Array` for mapped parameters
  - Constant parameters continue to receive scalar values
  - Output contract remains relaxed (can return `pa.Array`, `list`, or `numpy.ndarray`)
  - This ensures true vectorization performance with PyArrow compute functions

### Improved
- **SeqEngine batch optimization for DualNode**
  - Single-DualNode pipelines now use batch execution in `.map()` operations
  - Results in 1 batch call instead of N individual calls
  - Significant performance improvement even without DaftEngine
- **Error messages for DualNode**
  - Clear guidance when trying to batch non-convertible types (dataclasses, custom objects)
  - Recommends using regular `@node` decorator for complex types

### Removed
- Fallback logic that passed lists when PyArrow conversion failed
- This was causing inconsistent behavior between SeqEngine and DaftEngine

### Documentation
- Updated `docs/essentials/nodes.mdx` with comprehensive DualNode contract documentation
- Updated `docs/scaling/daft-engine.mdx` with PyArrow examples and requirements
- Updated `guides/DUAL_NODE_IMPLEMENTATION.md` with anti-patterns and best practices
- Added clear "When to Use" guidance: ✅ primitive types with vectorization, ❌ complex types

## [0.4.0] - 2025-11-19

### Changed
- **BREAKING**: Renamed `SequentialEngine` to `SeqEngine` for consistency and brevity
- **BREAKING**: Moved `cache` and `callbacks` parameters from `Pipeline` to `Engine` classes
  - Old: `Pipeline(nodes=[...], cache=..., callbacks=[...])`
  - New: `engine = SeqEngine(cache=..., callbacks=[...]); Pipeline(nodes=[...], engine=engine)`
  - This separates execution concerns from pipeline definition
  - All engines (SeqEngine, DaskEngine, DaftEngine) now support cache and callbacks uniformly

### Added
- `API_MIGRATION.md` guide to help migrate from old API to engine-centric architecture
- Shared execution orchestrator across all engines for consistency
- Engine compatibility checking for callbacks

### Deprecated
- `Pipeline.with_cache()` method (use `engine` parameter instead)
- `Pipeline.with_callbacks()` method (use `engine` parameter instead)

## [0.3.0] - 2025-11-19

### Added
- Initial release of HyperNodes
- Hierarchical, modular pipeline system for ML/AI workflows
- Node decorator for converting functions into pipeline nodes
- Pipeline class for composing nodes into DAGs
- Automatic dependency resolution based on function signatures
- Nested pipeline support (pipelines as nodes)
- Intelligent caching system with content-addressed signatures
- DiskCache implementation for persistent caching
- Multiple execution engines:
  - SeqEngine (default)
  - DaskEngine (parallel map operations)
  - DaftEngine (distributed DataFrames)
- Stateful objects support for expensive resources (models, DB connections)
- Progress tracking with ProgressCallback
- Distributed tracing support with TelemetryCallback
- Pipeline visualization with Graphviz
- Comprehensive test suite
- Documentation and examples

[0.1.0]: https://github.com/gilad-rubin/hypernodes/releases/tag/v0.1.0
