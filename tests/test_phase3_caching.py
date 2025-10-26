"""
Phase 3: Caching Tests

Test caching functionality with computation signatures and selective re-execution.
These tests validate core caching without callbacks (callbacks are Phase 4).
"""

import tempfile

from hypernodes import DiskCache, Pipeline, node

# Helper to track execution
execution_log = []


def setup_function():
    """Clear execution log before each test."""
    global execution_log
    execution_log = []


def test_3_1_single_node_cache_hit():
    """Test 3.1: Basic caching functionality.

    Validates:
    - Cache configured via cache=DiskCache(...) parameter
    - Cached results returned without re-execution
    - Different inputs cause cache miss
    """
    global execution_log

    with tempfile.TemporaryDirectory() as tmpdir:

        @node(output_name="result")
        def add_one(x: int) -> int:
            execution_log.append(f"add_one({x})")
            return x + 1

        # Configure pipeline with cache
        pipeline = Pipeline(nodes=[add_one], cache=DiskCache(path=tmpdir))

        # First run - should execute
        result1 = pipeline.run(inputs={"x": 5})
        assert result1 == {"result": 6}
        assert execution_log == ["add_one(5)"]

        # Second run with same input - should hit cache
        execution_log.clear()
        result2 = pipeline.run(inputs={"x": 5})
        assert result2 == {"result": 6}
        assert execution_log == []  # Not executed!

        # Third run with different input - should execute
        result3 = pipeline.run(inputs={"x": 10})
        assert result3 == {"result": 11}
        assert execution_log == ["add_one(10)"]


def test_3_2_partial_cache_hit_in_chain():
    """Test 3.2: Selective re-execution when some nodes cached.

    Validates:
    - Only changed nodes re-execute
    - Upstream cached results reused
    - Cache invalidation works correctly
    """
    global execution_log

    with tempfile.TemporaryDirectory() as tmpdir:

        @node(output_name="doubled")
        def double(x: int) -> int:
            execution_log.append(f"double({x})")
            return x * 2

        @node(output_name="result")
        def add_one(doubled: int) -> int:
            execution_log.append(f"add_one({doubled})")
            return doubled + 1

        pipeline = Pipeline(nodes=[double, add_one], cache=DiskCache(path=tmpdir))

        # First run - both execute
        result1 = pipeline.run(inputs={"x": 5})
        assert result1 == {"doubled": 10, "result": 11}
        assert execution_log == ["double(5)", "add_one(10)"]

        # Second run with same input - both cached
        execution_log.clear()
        result2 = pipeline.run(inputs={"x": 5})
        assert result2 == {"doubled": 10, "result": 11}
        assert execution_log == []  # Nothing executed!

        # Third run with different input - both execute
        result3 = pipeline.run(inputs={"x": 10})
        assert result3 == {"doubled": 20, "result": 21}
        assert execution_log == ["double(10)", "add_one(20)"]


def test_3_3_map_with_independent_item_caching():
    """Test 3.3: Each map item cached independently.

    Validates:
    - Map items cached independently
    - Cache hits across map calls
    - Only new items re-execute
    """
    global execution_log

    with tempfile.TemporaryDirectory() as tmpdir:

        @node(output_name="result")
        def add_one(x: int) -> int:
            execution_log.append(f"add_one({x})")
            return x + 1

        pipeline = Pipeline(nodes=[add_one], cache=DiskCache(path=tmpdir))

        # First run with [1, 2, 3] - all execute
        results1 = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
        assert results1 == {"result": [2, 3, 4]}
        assert execution_log == ["add_one(1)", "add_one(2)", "add_one(3)"]

        # Second run with same inputs - all cached
        execution_log.clear()
        results2 = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
        assert results2 == {"result": [2, 3, 4]}
        assert execution_log == []  # All cached!

        # Third run with [2, 3, 4] - only 4 executes (2 and 3 cached)
        results3 = pipeline.map(inputs={"x": [2, 3, 4]}, map_over="x")
        assert results3 == {"result": [3, 4, 5]}
        assert execution_log == ["add_one(4)"]  # Only new item!


def test_3_4_cache_disabled_for_specific_node():
    """Test 3.4: Node with cache=False always executes.

    Validates:
    - Nodes marked with cache=False never use cache
    - Even with identical inputs, node always executes
    - Downstream nodes can still be cached
    """
    global execution_log

    with tempfile.TemporaryDirectory() as tmpdir:

        @node(output_name="random_value", cache=False)
        def get_random(seed: int) -> int:
            execution_log.append(f"get_random({seed})")
            # Simulating non-deterministic behavior
            import random

            random.seed(seed)
            return random.randint(1, 100)

        @node(output_name="result")
        def add_one(random_value: int) -> int:
            execution_log.append(f"add_one({random_value})")
            return random_value + 1

        pipeline = Pipeline(nodes=[get_random, add_one], cache=DiskCache(path=tmpdir))

        # First run
        result1 = pipeline.run(inputs={"seed": 42})
        first_random = result1["random_value"]
        assert execution_log == ["get_random(42)", f"add_one({first_random})"]

        # Second run with same input - get_random ALWAYS executes (cache=False)
        # but add_one may be cached if random value is same
        execution_log.clear()
        result2 = pipeline.run(inputs={"seed": 42})
        assert result2["random_value"] == first_random  # Same random value (same seed)
        assert execution_log[0] == "get_random(42)"  # Always executes
        # Since seed is same, random value should be same, so add_one cached
        assert len(execution_log) == 1  # Only get_random executed


def test_3_5_code_change_invalidates_cache():
    """Test 3.5: Changing function code invalidates cache.

    Validates:
    - Function code changes detected
    - Cache invalidated when code changes
    - New results computed with updated code
    """
    global execution_log

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a function dynamically to simulate code change
        def make_adder(amount: int):
            @node(output_name="result")
            def add_x(x: int) -> int:
                execution_log.append(f"add_{amount}({x})")
                return x + amount

            return add_x

        # First version: add 1
        add_one = make_adder(1)
        pipeline = Pipeline(nodes=[add_one], cache=DiskCache(path=tmpdir))

        result1 = pipeline.run(inputs={"x": 5})
        assert result1 == {"result": 6}
        assert execution_log == ["add_1(5)"]

        # Run again - should cache
        execution_log.clear()
        result2 = pipeline.run(inputs={"x": 5})
        assert result2 == {"result": 6}
        assert result2 == result1  # Same result from cache
        assert execution_log == []

        # Create new pipeline with "modified" function (add 2 instead)
        execution_log.clear()
        add_two = make_adder(2)
        pipeline2 = Pipeline(nodes=[add_two], cache=DiskCache(path=tmpdir))

        # Should execute with new code (different function)
        result3 = pipeline2.run(inputs={"x": 5})
        assert result3 == {"result": 7}  # Different result!
        assert execution_log == ["add_2(5)"]


def test_3_6_diamond_pattern_with_cache():
    """Test 3.6: Cache works correctly with diamond dependency pattern.

    Validates:
    - Complex DAG patterns cached correctly
    - Each node cached independently
    - Cache revalidation follows dependency graph
    """
    global execution_log

    with tempfile.TemporaryDirectory() as tmpdir:

        @node(output_name="doubled")
        def double(x: int) -> int:
            execution_log.append(f"double({x})")
            return x * 2

        @node(output_name="tripled")
        def triple(x: int) -> int:
            execution_log.append(f"triple({x})")
            return x * 3

        @node(output_name="result")
        def add(doubled: int, tripled: int) -> int:
            execution_log.append(f"add({doubled},{tripled})")
            return doubled + tripled

        pipeline = Pipeline(nodes=[double, triple, add], cache=DiskCache(path=tmpdir))

        # First run - all execute
        result1 = pipeline.run(inputs={"x": 5})
        assert result1 == {"doubled": 10, "tripled": 15, "result": 25}
        assert execution_log == ["double(5)", "triple(5)", "add(10,15)"]

        # Second run - all cached
        execution_log.clear()
        result2 = pipeline.run(inputs={"x": 5})
        assert result2 == {"doubled": 10, "tripled": 15, "result": 25}
        assert execution_log == []

        # Third run with different input - all execute
        result3 = pipeline.run(inputs={"x": 10})
        assert result3 == {"doubled": 20, "tripled": 30, "result": 50}
        assert execution_log == ["double(10)", "triple(10)", "add(20,30)"]
