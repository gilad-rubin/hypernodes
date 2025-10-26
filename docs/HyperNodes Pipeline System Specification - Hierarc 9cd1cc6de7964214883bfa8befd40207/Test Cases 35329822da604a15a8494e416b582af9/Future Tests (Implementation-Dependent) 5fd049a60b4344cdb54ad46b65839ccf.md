# Future Tests (Implementation-Dependent)

# Phase 8: Backend-Specific Tests

**Note:** These tests depend on specific backend implementations.

## Test 8.1: Local Threaded Backend

- Verify parallel execution with threaded backend
- Track concurrent execution

## Test 8.2: Local Multiprocess Backend

- Verify multiprocess execution
- Ensure proper serialization of inputs/outputs

## Test 8.3: Modal Backend

- Verify remote execution on Modal
- Test context propagation
- Verify telemetry across boundaries

## Test 8.4: Nested with Different Backends

- Inner pipeline on Modal, outer local
- Verify results returned correctly
- Verify telemetry links across backends

---

# Phase 9: Telemetry Tests

## Test 9.1: Logfire Integration

- Verify LogfireCallback creates spans
- Check span hierarchy matches pipeline structure

## Test 9.2: Nested Pipeline Telemetry

- Verify nested pipeline spans are children of parent
- Check context propagation

## Test 9.3: Remote Execution Telemetry

- Verify spans link across local/remote boundaries
- Check metadata includes backend info

---

# Test Execution Strategy

**Recommended order:**

1. Run Phase 1 tests first (Core Execution)
2. Ensure all pass before moving to Phase 2
3. Progress through phases sequentially
4. Phase 8-9 tests require specific infrastructure

**For LLM Code Generation:**

Provide tests one phase at a time, ensuring each phase passes before implementing the next. This incremental approach helps catch design issues early and provides clear feedback on implementation quality.