# Dual-Mode Pattern Summary

**Problem:** "When I think in terms of lists, my brain explodes!"

**Solution:** Think singular, run batch automatically!

---

## The Core Insight

You've identified a real tension in data processing:

1. **For Understanding:** Singular functions are easy to reason about
2. **For Performance:** Batch operations are 10-100x faster

**The Dual-Mode Pattern resolves this!**

---

## How It Works

### Step 1: Define BOTH Versions

```python
from hypernodes.batch_adapter import batch_optimized

@batch_optimized
class EncodeText:
    '''Think singular, run batch!'''
    
    @staticmethod
    def singular(text: str, encoder) -> List[float]:
        """Easy to understand - ONE text!"""
        return encoder.encode(text)
    
    @staticmethod
    def batch(texts: List[str], encoder) -> List[List[float]]:
        """Optimized - MANY texts at once!"""
        return encoder.encode_batch(texts)  # 10-100x faster!
```

### Step 2: Use Singular in Pipeline

```python
from hypernodes import Pipeline, node

# Think about ONE item (easy!)
@node(output_name="embedding")
def encode_text(text: str, encoder) -> List[float]:
    return EncodeText.singular(text, encoder)

# Construct pipeline with singular logic
pipeline = Pipeline(nodes=[encode_text])
```

### Step 3: Run with Batch Performance

```python
# DaftEngine discovers batch version automatically
pipeline_with_engine = pipeline.with_engine(DaftEngine())

# When you map, it uses batch version behind the scenes!
results = pipeline.map(
    inputs={"text": ["text1", "text2", ..., "text1000"]},
    map_over="text"
)
# → Uses EncodeText.batch() for 10-100x speedup!
```

---

## Three Usage Patterns

### Pattern 1: Class-Based (Recommended)

Best for related operations (encoding, searching, etc.)

```python
@batch_optimized
class SearchIndex:
    @staticmethod
    def singular(query: str, index) -> List[dict]:
        """Think: Search ONE query"""
        return index.search(query, k=10)
    
    @staticmethod
    def batch(queries: List[str], index) -> List[List[dict]]:
        """Run: Search MANY queries at once"""
        return index.search_batch(queries, k=10)
```

### Pattern 2: Function-Based with Custom Batch

Best when you want to define batch separately

```python
@batch_optimized(auto_wrap=False)
def process_doc(doc: dict) -> dict:
    """Process ONE document (singular thinking)"""
    return {
        "id": doc["id"],
        "word_count": len(doc["text"].split())
    }

@process_doc.batch_version
def process_doc_batch(docs: List[dict]) -> List[dict]:
    """Process MANY documents (batch performance)"""
    return [process_doc(doc) for doc in docs]
```

### Pattern 3: Auto-Wrap (For Simple Cases)

Best for simple transformations where custom batch isn't needed

```python
@batch_optimized(auto_wrap=True)
def extract_metadata(doc: dict) -> dict:
    """Auto-creates batch version that loops"""
    return {
        "id": doc["id"],
        "length": len(doc["text"])
    }

# Batch version automatically created!
# extract_metadata.batch([doc1, doc2, ...]) works!
```

---

## Benefits Over Manual Batching

### ❌ Manual Batching (confusing)

```python
# You have to think in lists everywhere!

@node(output_name="embeddings")
def encode_texts(texts: List[str], encoder) -> List[List[float]]:
    """My brain: Wait, is this a list of texts or a list of lists?"""
    return encoder.encode_batch(texts)

# Debugging is hard:
# - What if texts is empty?
# - What if one text fails?
# - How do I test ONE text?
```

### ✅ Dual-Mode (clear)

```python
# Singular version (easy to understand!)

@batch_optimized
class EncodeText:
    @staticmethod
    def singular(text: str, encoder) -> List[float]:
        """ONE text → ONE embedding (crystal clear!)"""
        return encoder.encode(text)
    
    @staticmethod
    def batch(texts: List[str], encoder) -> List[List[float]]:
        """MANY texts → MANY embeddings (optimized!)"""
        return encoder.encode_batch(texts)

# Debugging is easy:
# - Test singular version with one text
# - Pipeline construction thinks singular
# - Performance uses batch automatically
```

---

## Real Example: Your Retrieval Pipeline

### Before (Hard to Reason About)

```python
# Current: Mapped pipeline nodes
encode_single_passage = Pipeline(nodes=[encode_passage])

encode_passages_mapped = encode_single_passage.as_node(
    input_mapping={"passages": "passage"},
    output_mapping={"encoded_passage": "encoded_passages"},
    map_over="passages",  # Brain: "Wait, passages becomes passage?"
    name="encode_passages_mapped"
)

# Mental overhead:
# - passages → passage (input mapping)
# - encoded_passage → encoded_passages (output mapping)
# - Have to track singular/plural everywhere!
```

### After (Easy to Think About)

```python
# Dual-mode: Think singular, run batch!

@batch_optimized
class EncodePassage:
    @staticmethod
    def singular(passage: dict, encoder) -> dict:
        """ONE passage → ONE encoded passage (clear!)"""
        embedding = encoder.encode(passage["text"])
        return {"uuid": passage["uuid"], "embedding": embedding}
    
    @staticmethod
    def batch(passages: List[dict], encoder) -> List[dict]:
        """MANY passages → MANY encoded passages (fast!)"""
        texts = [p["text"] for p in passages]
        embeddings = encoder.encode_batch(texts)
        return [
            {"uuid": p["uuid"], "embedding": emb}
            for p, emb in zip(passages, embeddings)
        ]

# Use in pipeline (think singular!)
@node(output_name="encoded_passage")
def encode_passage(passage: dict, encoder) -> dict:
    return EncodePassage.singular(passage, encoder)

# Or use batch directly (if you want)
@node(output_name="encoded_passages")
def encode_passages(passages: List[dict], encoder) -> List[dict]:
    return EncodePassage.batch(passages, encoder)

# Choose based on context!
```

---

## When to Use Each Mode

### Use Singular Version When:

✅ **Constructing pipeline** - easier to understand  
✅ **Debugging** - test with one item  
✅ **Unit testing** - simpler test cases  
✅ **Each item is different** - can't batch anyway  

### Use Batch Version When:

✅ **Encoding all passages** - vectorize for speed  
✅ **Processing uniform data** - all same operation  
✅ **Performance matters** - 10-100x speedup  
✅ **Library has *_batch()** - leverage it!  

---

## Integration with DaftEngine

DaftEngine can automatically discover and use batch versions:

```python
# Future: Automatic batch detection

@node(output_name="embedding")
def encode_text(text: str, encoder) -> List[float]:
    return EncodeText.singular(text, encoder)

# Attach batch version as metadata
encode_text._batch_version = EncodeText.batch
encode_text._is_batch_optimized = True

# DaftEngine discovers this and uses batch version when mapping!
pipeline = Pipeline(nodes=[encode_text], engine=DaftEngine())

# Internally:
# - map() detects _batch_version
# - Uses EncodeText.batch() instead of looping
# - 10-100x speedup automatically!
```

---

## Gradual Optimization Path

**Start simple, optimize incrementally!**

### Phase 1: Singular Only (Easy)

```python
@node(output_name="result")
def process_item(item: dict) -> dict:
    return {"id": item["id"], "processed": True}

# Easy to understand, works fine for small scale
```

### Phase 2: Add Auto-Wrap (Still Easy)

```python
@batch_optimized(auto_wrap=True)
class ProcessItem:
    @staticmethod
    def singular(item: dict) -> dict:
        return {"id": item["id"], "processed": True}

# Batch version auto-created (loops)
# Same performance, but batch API available
```

### Phase 3: Custom Batch (Optimized)

```python
@batch_optimized
class ProcessItem:
    @staticmethod
    def singular(item: dict) -> dict:
        return {"id": item["id"], "processed": True}
    
    @staticmethod
    def batch(items: List[dict]) -> List[dict]:
        # Vectorized implementation
        ids = [item["id"] for item in items]
        # Process all IDs at once (10-100x faster!)
        results = vectorized_process(ids)
        return [{"id": id, "processed": True} for id in ids]

# Now you have real performance gains!
```

---

## Summary: The Magic

**The dual-mode pattern gives you:**

1. ✅ **Singular thinking** - Easy on your brain
2. ✅ **Batch performance** - Fast execution  
3. ✅ **Gradual optimization** - Start simple, optimize later
4. ✅ **Best of both worlds** - Clarity AND speed

**No more "list explosion" in your head!**

---

## Files Created

1. **`src/hypernodes/batch_adapter.py`** - Dual-mode helper decorators
2. **`scripts/test_dual_mode_pattern.py`** - Examples and tests
3. **`scripts/test_exact_repro_OPTIMIZED.py`** - Your pipeline optimized
4. **`docs/OPTIMIZATION_GUIDE.md`** - Complete optimization guide
5. **`outputs/RETRIEVAL_OPTIMIZATION_RECOMMENDATIONS.md`** - Specific recommendations

---

**Bottom line:** You can keep thinking in terms of ONE item (easy!), but get performance of BATCH operations (fast!) automatically! 🚀

