#!/usr/bin/env python3
"""
CORRECT Usage of @daft.cls for Your Retrieval Pipeline

Based on Daft documentation (guides/daft-new-udf.md):

1. @daft.cls handles lazy initialization AUTOMATICALLY
   - __init__ not called immediately
   - Called once per worker when needed
   - No need for _ensure_loaded() pattern!

2. Use @daft.method.batch for batch methods
   - Tells Daft to use batch processing
   - Receives daft.Series, returns daft.Series
   - Can leverage vectorized operations
"""

import daft
from daft import DataType, Series
from typing import List, Any


# ==================== CORRECT Pattern: @daft.cls with Lazy Init ====================

@daft.cls(
    max_concurrency=2,  # Limit concurrent instances
    use_process=True,   # Process isolation for thread safety
)
class ColBERTEncoder:
    """
    Daft handles lazy initialization automatically!
    
    - __init__ is NOT called when you create: encoder = ColBERTEncoder("model-name")
    - __init__ IS called once per worker during execution
    - Instances are reused for multiple rows
    - NO need for _ensure_loaded() pattern!
    """
    
    def __init__(self, model_name: str, trust_remote_code: bool = True):
        """
        This is called ONCE per worker (lazy initialization).
        Daft handles this automatically - no manual lazy loading needed!
        """
        print(f"[Worker Init] Loading {model_name}...")
        # Load model directly - Daft ensures this happens once per worker
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"[Worker Init] Model loaded!")
    
    def encode(self, text: str, is_query: bool = False) -> List[float]:
        """Row-wise encoding (default method)."""
        # No _ensure_loaded() needed - __init__ already called by Daft!
        return self.model.encode(text).tolist()
    
    @daft.method.batch(return_dtype=DataType.python())
    def encode_batch(self, texts: Series, is_query: bool = False) -> Series:
        """
        Batch encoding - Daft knows to use this for batch processing!
        
        The @daft.method.batch decorator tells Daft:
        - This method processes entire batches (Series input/output)
        - Use this instead of calling encode() row-by-row
        - Enables vectorization for 10-100x speedup!
        """
        # Convert Series to Python list
        text_list = texts.to_pylist()
        
        # Batch encode (vectorized!)
        embeddings = self.model.encode(text_list)
        
        # Convert back to Series
        return Series.from_pylist([emb.tolist() for emb in embeddings])


# ==================== CORRECT Pattern: Multiple Instances ====================

@daft.cls
class PLAIDIndex:
    """Vector index with automatic lazy initialization."""
    
    def __init__(self, encoded_passages: List[dict], index_folder: str, index_name: str):
        """
        Called once per worker automatically.
        Build index from encoded passages.
        """
        print(f"[Worker Init] Building PLAID index with {len(encoded_passages)} passages...")
        # Build index - Daft ensures this happens once per worker
        self._documents = {p["uuid"]: p["embedding"] for p in encoded_passages}
        print(f"[Worker Init] Index built!")
    
    def search(self, query_embedding: List[float], k: int) -> List[dict]:
        """Row-wise search."""
        # No _ensure_loaded() - index already built by __init__!
        results = []
        for i, (doc_id, emb) in enumerate(list(self._documents.items())[:k]):
            results.append({"id": doc_id, "score": 1.0 / (i + 1)})
        return results
    
    @daft.method.batch(return_dtype=DataType.python())
    def search_batch(self, query_embeddings: Series, k: int) -> Series:
        """
        Batch search - process multiple queries at once!
        
        @daft.method.batch tells Daft to use this for batch processing.
        """
        queries = query_embeddings.to_pylist()
        results = [self.search(q, k) for q in queries]
        return Series.from_pylist(results)


# ==================== Usage Example ====================

def example_usage():
    """Show how @daft.cls works with lazy initialization."""
    
    print("\n" + "="*70)
    print("CORRECT @daft.cls USAGE")
    print("="*70)
    
    # Step 1: Create instance (NO __init__ called yet!)
    print("\n1. Creating ColBERTEncoder instance...")
    encoder = ColBERTEncoder("sentence-transformers/all-MiniLM-L6-v2")
    print("   ✓ Instance created (model NOT loaded yet - lazy!)")
    
    # Step 2: Use in DataFrame (Daft calls __init__ on workers)
    print("\n2. Using in Daft DataFrame...")
    df = daft.from_pydict({"text": ["hello", "world", "test"]})
    
    # Option A: Row-wise (default)
    print("   a) Row-wise encoding...")
    df_rowwise = df.with_column("embedding", encoder.encode(daft.col("text"), is_query=False))
    result = df_rowwise.collect()
    print(f"   ✓ Encoded {len(result)} texts (row-wise)")
    
    # Option B: Batch method (FASTER!)
    print("   b) Batch encoding...")
    df_batch = df.with_column("embedding", encoder.encode_batch(daft.col("text"), is_query=False))
    result = df_batch.collect()
    print(f"   ✓ Encoded {len(result)} texts (batch - faster!)")
    
    print("\n3. How it worked:")
    print("   - Daft saved encoder config when you created the instance")
    print("   - During collect(), Daft called __init__ on each worker")
    print("   - Model loaded ONCE per worker, reused for all rows")
    print("   - NO manual lazy loading needed!")


# ==================== CORRECT Optimization for Retrieval Pipeline ====================

def correct_optimization_pattern():
    """Show the CORRECT way to optimize the retrieval pipeline."""
    
    print("\n" + "="*70)
    print("CORRECT OPTIMIZATION PATTERN")
    print("="*70)
    
    print("""
RECOMMENDED APPROACH for your retrieval pipeline:

1. ✅ Use @daft.cls (Daft handles lazy init automatically!)
2. ✅ Use @daft.method.batch for batch methods
3. ✅ NO _ensure_loaded() pattern needed!

Example:

@daft.cls(max_concurrency=2, use_process=True)
class ColBERTEncoder:
    '''Daft handles lazy initialization!'''
    
    def __init__(self, model_name: str):
        # This is called ONCE per worker automatically
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def encode(self, text: str, is_query: bool = False) -> List[float]:
        '''Row-wise method.'''
        return self.model.encode(text).tolist()
    
    @daft.method.batch(return_dtype=DataType.python())
    def encode_batch(self, texts: Series, is_query: bool = False) -> Series:
        '''Batch method - Daft uses this for batch processing!'''
        text_list = texts.to_pylist()
        embeddings = self.model.encode(text_list)
        return Series.from_pylist([emb.tolist() for emb in embeddings])

# Usage:
encoder = ColBERTEncoder("my-model")  # Config saved, __init__ NOT called yet

df = daft.from_pydict({"text": ["hello", "world"]})

# Daft calls __init__ on workers, then uses encode_batch automatically!
df = df.with_column("embedding", encoder.encode_batch(daft.col("text")))
result = df.collect()
    """)


if __name__ == "__main__":
    example_usage()
    correct_optimization_pattern()

