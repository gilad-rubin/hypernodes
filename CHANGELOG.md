# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
