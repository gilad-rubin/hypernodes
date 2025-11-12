Modal/Daft Serialization Issue – Handoff Notes

Overview
Failure: uv run modal run scripts/test_modal_failure_repro.py (from repo root) ends with ModuleNotFoundError: No module named 'test_modal_failure_repro' thrown by Daft’s UDF worker.
Works When: Running the same script from inside scripts/ or when Daft never emits UDFs referencing the script module.
Scope: Any pipeline that passes “script-defined” objects (stateful helpers, Pydantic models, nested structures) into DaftEngine while the script isn’t importable on the worker (Modal) will hit the same crash.
What Reproduces the Failure
Script: scripts/test_modal_failure_repro.py – lightweight mock of the Hebrew retrieval pipeline (encoders, RRFFusion, recall evaluator, nested pipelines via .as_node, etc.).
Command: uv run modal run scripts/test_modal_failure_repro.py from repo root.
Logs: Daft debug prints show recall_evaluator is the last captured stateful param before the worker panics. The worker stdout always ends with ModuleNotFoundError: No module named 'test_modal_failure_repro'.
Observation: Running the script after cd scripts succeeds, confirming the failure is tied to the script’s import path on Modal.
Anatomy of the Problem
Stateful Inputs: Daft auto-captures constant inputs (encoders, evaluators) and wraps them via @daft.cls. If the object’s class lives in a non-importable module (like scripts), cloudpickle’s default behavior tries to import that module inside workers and fails.
Data Row Objects: Even after capturing stateful params, outputs (e.g., Prediction, GroundTruth, dictionaries of models) can still carry references to the script module when they’re materialized into Daft DataFrames.
Module Registry: _make_class_serializable_by_value rewrites class definitions so __module__ = "__main__" and rebinds them inside the original module, but anything created before that rewrite—or nested within other objects—can slip through if not processed.
Changes Already Made
Added TDD coverage in tests/test_daft_modal_script_pattern.py:
test_unhinted_script_stateful_object_is_captured
test_should_capture_stateful_input_for_unimportable_classes
test_make_class_serializable_rebinds_module
Updated DaftEngine:
_should_capture_stateful_input now treats all non-primitive constants as stateful (ensures objects like RecallEvaluator are captured even without hints).
_make_class_serializable_by_value rebinds the new __main__ class into the original module to keep global references aligned.
_fix_output_tree recursively prepares returned values so nested structures get the by-value treatment before crossing process boundaries.
Despite these improvements, Modal runs from repo root still fail, meaning some remaining object(s) (likely row data or Pydantic instances) keep their original module metadata.

What Still Needs Investigation
Identify Remaining Offenders
Inspect the Daft columnar data (perhaps by logging stateful_inputs, DataFrame schema, or serializing per-node payloads) to see which outputs still mention test_modal_failure_repro.
Focus on Pydantic models produced after the stateful wrappers (e.g., Prediction, GroundTruth, nested dicts).
Ensure _fix_output_tree Is Applied Everywhere
Verify that every pipeline output path (including .map, nested pipelines, and intermediate .with_column) passes through _fix_output_tree before Daft serializes the batch to workers.
Consider Worker PYTHONPATH Fallback
As a short-term fail-safe, mount the scripts/ directory into Modal’s image and add it to PYTHONPATH. This shouldn’t be necessary once serialization is rock solid, but it can unblock production runs.
Audit Code Paths for Early Object Creation
Some script-defined objects may be instantiated before _make_class_serializable_by_value has a chance to run (e.g., module-level constants). Those need either manual fixing or a module import hook.
Suggested Next Actions
Add Instrumentation:
Log each stateful value before/after _prepare_stateful_value_for_daft.
Dump the types seen in _convert_output_value to confirm they’ve been rewritten to __main__.
Mini Harness Without Modal:
Run the script locally (no Modal) with DaftEngine(debug=True) to trigger the same path. The failure will show up as RecursionError (which Daft wraps into ModuleNotFoundError), making it easier to inspect objects.
Pytest Repro:
Consider a new test in tests/test_daft_modal_script_pattern.py that mimics the full pipeline with nested Pydantic outputs, ensuring the serialized rows no longer reference the script module.
Doc Updates / Fallback:
Document that until a full fix lands, cd scripts (or adding scripts/ to PYTHONPATH) is a workaround.
Files Involved
scripts/test_modal_failure_repro.py
src/hypernodes/integrations/daft/engine.py
tests/test_daft_modal_script_pattern.py
