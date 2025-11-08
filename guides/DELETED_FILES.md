# Deleted Files - Orchestrator/Engine Refactoring

These files were deleted during the orchestrator→engine merge refactoring on November 7, 2025.
All files were moved to system trash using the `trash` command and can be recovered if needed.

## Files Deleted

### 1. src/hypernodes/orchestrator.py (~240 lines)
**Deleted**: November 7, 2025
**Reason**: Merged into `HypernodesEngine._execute_pipeline()` method
**Replacement**: `src/hypernodes/engine.py` (lines 148-319)

**What it contained**:
- `class PipelineOrchestrator` - Node-by-node pipeline execution orchestrator
- Dependency resolution logic
- Parallel execution with futures
- Callback management

**Why deleted**: Orchestrator was an unnecessary abstraction. Each engine should implement its own orchestration strategy.

---

### 2. src/hypernodes/backend.py (~2000 lines)
**Deleted**: November 7, 2025
**Reason**: Old Backend architecture replaced by Engine architecture
**Replacement**: `src/hypernodes/engine.py` + modal migration needed

**What it contained**:
- `class Backend` - Abstract base class for execution backends
- `class LocalBackend` - Local execution backend (replaced by HypernodesEngine)
- `class ModalBackend` - Remote Modal execution backend (NOT YET MIGRATED)
- `class PipelineExecutionEngine` - Internal execution engine

**Status**:
- ✅ LocalBackend → migrated to HypernodesEngine
- ❌ ModalBackend → needs migration to ModalEngine (future work)

**Recover if**: You need to migrate ModalBackend to ModalEngine

---

### 3. src/hypernodes/executors/base.py (~82 lines)
**Deleted**: November 7, 2025
**Reason**: Old Engine ABC replaced by new Engine in engine.py
**Replacement**: `src/hypernodes/engine.py` (lines 27-76)

**What it contained**:
- `class Engine` - Abstract base class for execution engines
- Method signatures for `run()` and `map()`

**Why deleted**: New Engine ABC has cleaner interface with `_ctx` parameter convention.

---

### 4. src/hypernodes/executors/local.py (~1564 lines)
**Deleted**: November 7, 2025
**Reason**: Old HyperNodesEngine implementation replaced by new HypernodesEngine
**Replacement**: `src/hypernodes/engine.py` (HypernodesEngine class)

**What it contained**:
- `class HyperNodesEngine` - Old implementation with if-statement dispatching
- Sequential, async, threaded, parallel execution modes
- Map execution logic
- Lots of code duplication

**Why deleted**: New implementation uses strategy pattern with executor adapters, eliminating duplication.

---

## Total Deleted

- **Lines removed**: ~3,886 lines
- **Lines added**: ~950 lines
- **Net reduction**: 76% code reduction

## Recovery Instructions

If you need to recover any of these files:

### macOS/Linux:
```bash
# Files are in system trash, can be recovered via:
# 1. Open Trash/Bin in file manager
# 2. Search for filename
# 3. Right-click → Restore

# Or if you have trash-cli:
trash-list | grep backend.py
trash-restore
```

### Manual Recovery:
The files were deleted on November 7, 2025. Check your system trash for:
- `backend.py`
- `orchestrator.py`
- `executors/base.py`
- `executors/local.py`

## Alternative: Git History

All these files exist in git history before the refactoring:
```bash
# Find the last commit before deletion
git log --all --full-history -- src/hypernodes/backend.py

# Restore from git history
git checkout <commit-hash> -- src/hypernodes/backend.py
```

## Archival Recommendation

If you want to keep these files for reference, create an archive:
```bash
mkdir -p old/src/hypernodes/executors

# Restore from trash or git, then:
mv backend.py old/src/hypernodes/
mv orchestrator.py old/src/hypernodes/
mv executors/base.py old/src/hypernodes/executors/
mv executors/local.py old/src/hypernodes/executors/
```

## What's NOT Deleted

These files remain and were updated to use the new Engine architecture:
- ✅ `src/hypernodes/engine.py` - New Engine ABC + HypernodesEngine
- ✅ `src/hypernodes/executor_adapters.py` - Executor strategy pattern
- ✅ `src/hypernodes/node_execution.py` - Node execution logic
- ✅ `src/hypernodes/pipeline.py` - Updated to use Engine
- ✅ `src/hypernodes/daft_backend.py` - Updated to inherit from Engine
- ✅ `src/hypernodes/executors/__init__.py` - Updated exports
