
# Current Architecture Analysis

## 1. SequentialEngine (`src/hypernodes/sequential_engine.py`)
- **Responsibility**: Executes nodes in topological order one by one.
- **Caching**: Currently delegates node execution to `node_execution.py`? Or handles it internally? 
- **Callbacks**: Needs to fire `on_node_start`, `on_node_end`, `on_pipeline_start`, `on_pipeline_end`.

## 2. DaftEngine (`src/hypernodes/integrations/daft/engine.py`)
- **Responsibility**: Translates the pipeline into a Daft plan and executes it.
- **Caching**: Handles caching logic *internally* within its `run` loop (lines 120-152 in grep).
  - Checks cache -> Joins if hit.
  - Executes -> Writes to cache if miss.
- **Callbacks**:
  - Fires `notify_pipeline_start/end`.
  - Bridges Daft events to HyperNodes callbacks via `HyperNodesDaftSubscriber`.
  - Fires `notify_node_cached`.

## 3. Issues & Discrepancies
1.  **Duplication**: Caching logic seems to be re-implemented in engines. `SequentialEngine` likely has its own (or delegates to `node_execution.py`), while `DaftEngine` has its own embedded logic.
2.  **Inconsistency**: If a new engine is added, we have to re-implement caching and callback firing.
3.  **Coupling**: `DaftEngine` depends directly on `CallbackDispatcher` and `Cache` behavior.

## Goal
Unify the "outer loop" or "execution harness" so that caching and callbacks are handled consistently, or provide clear utilities so engines don't reinvent the wheel.

