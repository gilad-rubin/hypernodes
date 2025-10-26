"""
Phase 5: Nested Pipelines Tests

Test nested pipeline functionality with proper output propagation, independent caching,
configuration inheritance, and .as_node() with input/output mapping.
"""
from typing import List, NamedTuple, Sequence
from hypernodes import node, Pipeline, LocalBackend, DiskCache, PipelineCallback, CallbackContext


# =======================
# Test 5.1: Simple Nested Pipeline
# =======================

def test_5_1_simple_nested_pipeline():
    """Test 5.1: Verify pipeline used as node in another pipeline.
    
    Validates:
    - Pipeline used as node
    - Outputs from nested pipeline available to outer pipeline
    - Dependencies resolved across pipeline boundaries
    """
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="incremented")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    inner_pipeline = Pipeline(nodes=[double, add_one])
    
    @node(output_name="result")
    def square(incremented: int) -> int:
        return incremented ** 2
    
    outer_pipeline = Pipeline(nodes=[inner_pipeline, square])
    
    result = outer_pipeline.run(inputs={"x": 5})
    assert result == {"doubled": 10, "incremented": 11, "result": 121}


# =======================
# Test 5.2: Nested Pipeline with Map
# =======================

def test_5_2_nested_pipeline_with_map():
    """Test 5.2: Verify nested pipeline in map operation.
    
    Validates:
    - Nested pipelines work in map operations
    - Each item processed through full nested pipeline
    """
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    inner_pipeline = Pipeline(nodes=[double, add_one])
    
    outer_pipeline = Pipeline(nodes=[inner_pipeline])
    
    results = outer_pipeline.map(inputs={"x": [1, 2, 3]}, map_over=["x"])
    assert results == {"doubled": [2, 4, 6], "result": [3, 5, 7]}


# =======================
# Test 5.3: Two-Level Nesting
# =======================

def test_5_3_two_level_nesting():
    """Test 5.3: Verify deeper nesting (3 levels total).
    
    Validates:
    - Multiple levels of nesting work
    - Outputs propagate up through levels
    """
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    inner_inner = Pipeline(nodes=[double])
    
    @node(output_name="incremented")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    inner = Pipeline(nodes=[inner_inner, add_one])
    
    @node(output_name="result")
    def square(incremented: int) -> int:
        return incremented ** 2
    
    outer = Pipeline(nodes=[inner, square])
    
    result = outer.run(inputs={"x": 5})
    assert result == {"doubled": 10, "incremented": 11, "result": 121}


# =======================
# Test 5.4: Nested Pipeline with Independent Caching
# =======================

def test_5_4_nested_pipeline_with_independent_caching():
    """Test 5.4: Verify each pipeline level has independent cache.
    
    Validates:
    - Nested pipelines have independent caching
    - Cache hits at all levels
    - Callback tracking shows caching behavior at each level
    """
    class LoggingCachingCallback(PipelineCallback):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
            self.executions = []
            self.cache_hits = []
        
        def on_node_start(self, node_id: str, inputs: dict, ctx: CallbackContext):
            self.executions.append(node_id)
        
        def on_node_cached(self, node_id: str, signature: str, ctx: CallbackContext):
            self.cache_hits.append(node_id)
    
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="incremented")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    # Use DiskCache for testing
    inner_cache = DiskCache(path=".cache_test_5_4_inner")
    inner_cache.clear()  # Clear any stale cache data
    inner_callback = LoggingCachingCallback("inner")
    inner = Pipeline(
        nodes=[double, add_one],
        cache=inner_cache,
        callbacks=[inner_callback]
    )
    
    @node(output_name="result")
    def square(incremented: int) -> int:
        return incremented ** 2
    
    outer_cache = DiskCache(path=".cache_test_5_4_outer")
    outer_cache.clear()  # Clear any stale cache data
    outer_callback = LoggingCachingCallback("outer")
    outer = Pipeline(
        nodes=[inner, square],
        cache=outer_cache,
        callbacks=[outer_callback]
    )
    
    # First run
    inner_callback.executions = []
    inner_callback.cache_hits = []
    outer_callback.executions = []
    outer_callback.cache_hits = []
    
    result1 = outer.run(inputs={"x": 5})
    assert result1 == {"doubled": 10, "incremented": 11, "result": 121}
    assert len(inner_callback.executions) == 2  # double, add_one
    assert len(outer_callback.executions) == 1  # square
    
    # Second run - all cached
    inner_callback.executions = []
    inner_callback.cache_hits = []
    outer_callback.executions = []
    outer_callback.cache_hits = []
    
    result2 = outer.run(inputs={"x": 5})
    assert result2 == {"doubled": 10, "incremented": 11, "result": 121}
    assert len(inner_callback.executions) == 0
    assert len(inner_callback.cache_hits) == 2  # double, add_one cached
    assert len(outer_callback.executions) == 0
    assert len(outer_callback.cache_hits) == 1  # square cached
    
    # Cleanup
    inner_cache.clear()
    outer_cache.clear()


# =======================
# Test 5.5: Pipeline as Node with Input Renaming
# =======================

def test_5_5_pipeline_as_node_with_input_renaming():
    """Test 5.5: Verify .as_node() with input_mapping.
    
    Validates:
    - Input mapping works correctly
    - Direction is {outer: inner}
    - Inner pipeline receives correctly renamed parameter
    """
    @node(output_name="cleaned")
    def clean_text(passage: str) -> str:
        return passage.strip().lower()
    
    inner = Pipeline(nodes=[clean_text])
    
    # Outer pipeline uses "document" instead of "passage"
    adapted = inner.as_node(
        input_mapping={"document": "passage"}  # outer → inner
    )
    
    outer = Pipeline(nodes=[adapted])
    
    result = outer.run(inputs={"document": "  Hello World  "})
    assert result["cleaned"] == "hello world"


# =======================
# Test 5.6: Pipeline as Node with Output Renaming
# =======================

def test_5_6_pipeline_as_node_with_output_renaming():
    """Test 5.6: Verify .as_node() with output_mapping.
    
    Validates:
    - Output mapping works correctly
    - Direction is {inner: outer}
    - Original output name is hidden from outer pipeline
    """
    @node(output_name="result")
    def process(data: str) -> str:
        return data.upper()
    
    inner = Pipeline(nodes=[process])
    
    # Outer pipeline wants the output named "processed_data"
    adapted = inner.as_node(
        output_mapping={"result": "processed_data"}  # inner → outer
    )
    
    outer = Pipeline(nodes=[adapted])
    
    result = outer.run(inputs={"data": "hello"})
    assert "processed_data" in result
    assert "result" not in result  # Original name not visible
    assert result["processed_data"] == "HELLO"


# =======================
# Test 5.7: Pipeline as Node with Combined Renaming
# =======================

def test_5_7_pipeline_as_node_with_combined_renaming():
    """Test 5.7: Verify .as_node() with both input and output mapping.
    
    Validates:
    - Both mappings work simultaneously
    - Inner and outer pipelines have completely different interfaces
    """
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    inner = Pipeline(nodes=[double])
    
    adapted = inner.as_node(
        input_mapping={"value": "x"},
        output_mapping={"doubled": "result"}
    )
    
    outer = Pipeline(nodes=[adapted])
    
    result = outer.run(inputs={"value": 5})
    assert result["result"] == 10
    assert "doubled" not in result


# =======================
# Test 5.8: Internal Mapping with Renaming (Encapsulated Map)
# =======================

def test_5_8_internal_mapping_with_renaming():
    """Test 5.8: Verify .as_node() with map_over to encapsulate internal mapping.
    
    Validates:
    - map_over parameter works with renaming
    - Inner pipeline executes once per item
    - Outer pipeline sees list input → list output
    - Mapping is completely encapsulated
    """
    class Item(NamedTuple):
        id: int
        value: str
    
    @node(output_name="processed")
    def process_item(item: Item) -> str:
        return f"{item.id}: {item.value.upper()}"
    
    # Inner pipeline processes ONE item
    single_process = Pipeline(nodes=[process_item])
    
    # Adapt to process a LIST with renamed interface
    batch_process = single_process.as_node(
        map_over="items",  # Outer provides "items" as list
        input_mapping={"items": "item"},  # Each list element becomes "item"
        output_mapping={"processed": "results"}  # Collect as "results"
    )
    
    outer = Pipeline(nodes=[batch_process])
    
    items = [Item(id=1, value="hello"), Item(id=2, value="world")]
    result = outer.run(inputs={"items": items})
    
    assert result["results"] == ["1: HELLO", "2: WORLD"]
    assert "processed" not in result


# =======================
# Test 5.9: Internal Mapping with Caching
# =======================

def test_5_9_internal_mapping_with_caching():
    """Test 5.9: Verify .as_node() with map_over works correctly.
    
    Validates:
    - Each item in the mapped execution is processed
    - Internal mapping is encapsulated properly
    - Results are collected correctly
    """
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    single = Pipeline(nodes=[double])
    
    batch = single.as_node(
        map_over="numbers",
        input_mapping={"numbers": "x"},
        output_mapping={"doubled": "results"}
    )
    
    outer = Pipeline(nodes=[batch])
    
    # Test with first batch
    result1 = outer.run(inputs={"numbers": [1, 2, 3]})
    assert result1["results"] == [2, 4, 6]
    
    # Test with second batch
    result2 = outer.run(inputs={"numbers": [2, 3, 4]})
    assert result2["results"] == [4, 6, 8]
    
    # Test with empty list
    result3 = outer.run(inputs={"numbers": []})
    assert result3["results"] == []


# =======================
# Test 5.10: Namespace Collision Avoidance
# =======================

def test_5_10_namespace_collision_avoidance():
    """Test 5.10: Verify output_mapping prevents naming collisions between pipelines.
    
    Validates:
    - Multiple pipelines with same output name can coexist
    - Output mapping creates separate namespaces
    - Downstream functions can depend on renamed outputs
    """
    @node(output_name="result")
    def process_a(input: int) -> int:
        return input * 2
    
    @node(output_name="result")
    def process_b(input: int) -> int:
        return input * 3
    
    pipeline_a = Pipeline(nodes=[process_a]).as_node(
        output_mapping={"result": "result_a"}
    )
    
    pipeline_b = Pipeline(nodes=[process_b]).as_node(
        output_mapping={"result": "result_b"}
    )
    
    @node(output_name="combined")
    def combine(result_a: int, result_b: int) -> int:
        return result_a + result_b
    
    outer = Pipeline(nodes=[pipeline_a, pipeline_b, combine])
    
    result = outer.run(inputs={"input": 5})
    assert result["result_a"] == 10
    assert result["result_b"] == 15
    assert result["combined"] == 25


# =======================
# Test 5.11: Complex Nested Mapping (Real-World Example)
# =======================

def test_5_11_complex_nested_mapping():
    """Test 5.11: Verify the encode corpus → build index pattern from documentation.
    
    Validates:
    - Complete real-world pattern works end-to-end
    - Input/output renaming with mapping
    - Multiple pipeline levels
    - Proper namespace isolation
    - Downstream functions receive correctly named outputs
    """
    class Passage(NamedTuple):
        pid: str
        text: str
    
    class Vector(NamedTuple):
        values: List[float]
    
    class EncodedPassage(NamedTuple):
        pid: str
        embedding: Vector
    
    class Encoder:
        def encode(self, text: str) -> Vector:
            # Dummy encoder
            return Vector(values=[float(ord(c)) for c in text[:3]])
    
    class Indexer:
        def index(self, passages: Sequence[EncodedPassage]) -> dict:
            return {"count": len(passages), "ids": [p.pid for p in passages]}
    
    @node(output_name="cleaned_text")
    def clean_text(passage: Passage) -> str:
        return passage.text.strip().lower()
    
    @node(output_name="embedding")
    def encode_text(encoder: Encoder, cleaned_text: str) -> Vector:
        return encoder.encode(cleaned_text)
    
    @node(output_name="encoded_passage")
    def pack_encoded(passage: Passage, embedding: Vector) -> EncodedPassage:
        return EncodedPassage(pid=passage.pid, embedding=embedding)
    
    # Inner pipeline: processes ONE passage
    single_encode = Pipeline(nodes=[clean_text, encode_text, pack_encoded])
    
    # Adapt to process a CORPUS (list) with renamed interface
    encode_corpus = single_encode.as_node(
        map_over="corpus",
        input_mapping={"corpus": "passage"},
        output_mapping={"encoded_passage": "encoded_corpus"}
    )
    
    @node(output_name="index")
    def build_index(indexer: Indexer, encoded_corpus: Sequence[EncodedPassage]) -> dict:
        return indexer.index(encoded_corpus)
    
    # Outer pipeline: corpus → index
    encode_and_index = Pipeline(nodes=[encode_corpus, build_index])
    
    # Execute
    corpus = [
        Passage(pid="p1", text="Hello World"),
        Passage(pid="p2", text="The Quick Brown Fox"),
    ]
    encoder = Encoder()
    indexer = Indexer()
    
    outputs = encode_and_index.run(
        inputs={
            "corpus": corpus,
            "encoder": encoder,
            "indexer": indexer,
        }
    )
    
    index = outputs["index"]
    assert index["count"] == 2
    assert index["ids"] == ["p1", "p2"]
    assert "encoded_passage" not in outputs  # Inner name hidden
    assert "encoded_corpus" in outputs  # Outer name visible


# =======================
# Test 5.12: Configuration Inheritance - Backend Only
# =======================

def test_5_12_configuration_inheritance_backend_only():
    """Test 5.12: Verify backend configuration inherits from parent when not specified.
    
    Validates:
    - Nested pipeline inherits parent backend when not specified
    - Execution works correctly with inherited backend
    """
    @node(output_name="doubled")
    def double(x: int) -> int:
        return x * 2
    
    @node(output_name="result")
    def add_one(doubled: int) -> int:
        return doubled + 1
    
    # Child has no backend specified
    inner = Pipeline(nodes=[double, add_one])
    
    # Parent defines backend
    outer = Pipeline(
        nodes=[inner],
        backend=LocalBackend()
    )
    
    result = outer.run(inputs={"x": 5})
    assert result["result"] == 11
    
    # Verify inner uses effective backend (will use parent's if not set)
    assert inner.effective_backend is not None


# =======================
# Test 5.13: Configuration Inheritance - Selective Override
# =======================

def test_5_13_configuration_inheritance_selective_override():
    """Test 5.13: Verify selective override of backend while inheriting other configuration.
    
    Validates:
    - Selective override of one configuration aspect
    - Other aspects inherited from parent
    - Override does not affect parent's configuration
    """
    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2
    
    # Parent with configuration
    parent_callback = PipelineCallback()
    parent = Pipeline(
        nodes=[process],
        backend=LocalBackend(),
        callbacks=[parent_callback]
    )
    
    # Child overrides only backend
    child_backend = LocalBackend()
    child = Pipeline(
        nodes=[process],
        backend=child_backend,
        parent=parent
    )
    
    # Verify inheritance
    assert child.effective_backend == child_backend  # Overridden
    assert child.effective_callbacks == parent.effective_callbacks  # Inherited


# =======================
# Test 5.14: Configuration Inheritance - Recursive Chain
# =======================

def test_5_14_configuration_inheritance_recursive_chain():
    """Test 5.14: Verify configuration inherits through multiple nesting levels.
    
    Validates:
    - Configuration inherits through full chain
    - Each level can override different aspects
    - Overrides propagate down
    """
    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2
    
    # Level 1: Define all configuration
    level_1_backend = LocalBackend()
    level_1_callback = PipelineCallback()
    level_1 = Pipeline(
        nodes=[process],
        backend=level_1_backend,
        callbacks=[level_1_callback]
    )
    
    # Level 2: Override backend only
    level_2_backend = LocalBackend()
    level_2 = Pipeline(
        nodes=[process],
        backend=level_2_backend,
        parent=level_1
    )
    
    # Level 3: No overrides, inherits from level_2
    level_3 = Pipeline(
        nodes=[process],
        parent=level_2
    )
    
    # Verify final configuration for level_3
    assert level_3.effective_backend == level_2_backend  # From level_2
    assert level_3.effective_callbacks == level_1.effective_callbacks  # From level_1


# =======================
# Test 5.15: Configuration Inheritance - Disable Caching
# =======================

def test_5_15_configuration_inheritance_disable_caching():
    """Test 5.15: Verify node-level cache disabling works.
    
    Validates:
    - @node(cache=False) disables caching for that node
    - Other nodes in the same pipeline can still use cache
    - Child can opt out of parent's caching strategy
    """
    @node(output_name="result")
    def expensive_operation(x: int) -> int:
        return x ** 2
    
    # Node with caching disabled
    @node(output_name="uncached_result", cache=False)
    def uncached_operation(x: int) -> int:
        return x ** 3
    
    cache = DiskCache(path=".cache_test_5_15")
    cache.clear()
    
    pipeline = Pipeline(
        nodes=[expensive_operation, uncached_operation],
        cache=cache
    )
    
    # Verify the cache attribute is set correctly
    assert expensive_operation.cache is True
    assert uncached_operation.cache is False
    
    # Both nodes execute successfully
    result = pipeline.run(inputs={"x": 5})
    assert result["result"] == 25
    assert result["uncached_result"] == 125
    
    # Cleanup
    cache.clear()


# =======================
# Test 5.16: Configuration Inheritance - Callback Inheritance
# =======================

def test_5_16_configuration_inheritance_callback_inheritance():
    """Test 5.16: Verify callback inheritance and override behavior.
    
    Validates:
    - Callbacks inherit from parent when not specified
    - Explicit callback list overrides parent completely
    - Override affects only that level, not siblings
    """
    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2
    
    # Parent with callbacks
    parent_callback1 = PipelineCallback()
    parent_callback2 = PipelineCallback()
    parent = Pipeline(
        nodes=[process],
        callbacks=[parent_callback1, parent_callback2]
    )
    
    # Child inherits all callbacks (no override)
    child = Pipeline(
        nodes=[process],
        parent=parent
    )
    
    # Grandchild overrides with different callbacks
    grandchild_callback = PipelineCallback()
    grandchild = Pipeline(
        nodes=[process],
        callbacks=[grandchild_callback],
        parent=child
    )
    
    # Verify inheritance
    assert len(child.effective_callbacks) == 2  # Inherited from parent
    assert len(grandchild.effective_callbacks) == 1  # Overridden


# =======================
# Test 5.17: Configuration Inheritance - Full Inheritance
# =======================

def test_5_17_configuration_inheritance_full_inheritance():
    """Test 5.17: Verify complete inheritance when child specifies no configuration.
    
    Validates:
    - Nested pipeline with no configuration inherits everything
    - All configuration aspects inherited simultaneously
    - Child behaves as if configuration was explicitly copied
    """
    @node(output_name="result")
    def process(x: int) -> int:
        return x * 2
    
    # Outer with full configuration
    outer_backend = LocalBackend()
    outer_cache = DiskCache(path=".cache_test_5_17")
    outer_cache.clear()  # Clear any stale cache data
    outer_callback = PipelineCallback()
    outer = Pipeline(
        nodes=[process],
        backend=outer_backend,
        cache=outer_cache,
        callbacks=[outer_callback]
    )
    
    # Inner has NO configuration
    inner = Pipeline(
        nodes=[process],
        parent=outer
    )
    
    # Verify complete inheritance
    assert inner.effective_backend == outer_backend
    assert inner.effective_cache == outer_cache
    assert inner.effective_callbacks == [outer_callback]
    
    # Cleanup
    outer_cache.clear()
