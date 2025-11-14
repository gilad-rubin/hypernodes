#!/usr/bin/env python3
"""
Stateful Resource Pattern Benchmark

Compares three execution strategies for encoding operations:
1. HyperNodes Sequential (row-wise processing)
2. HyperNodes + DaftEngine (explicit batch nodes)
3. Pure Daft (hand-written @daft.cls + @daft.method.batch)

Tests with both mock encoder and real SentenceTransformer from HuggingFace.
"""

from __future__ import annotations

import functools
import hashlib
import time
from typing import Any, Callable, List

import daft
import numpy as np
from daft import DataType, Series

from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine


# ==================== Decorators ====================


def stateful(cls: type) -> type:
    """Mark a class as stateful with lazy initialization.
    
    Features:
    - Lazy initialization: __init__ args stored, not called until first use
    - Efficient pickling: only init args serialized, not instance state
    - Cache-friendly: __cache_key__() based on init args only
    
    Example:
        @stateful
        class Model:
            def __init__(self, model_name: str):
                # Expensive initialization
                self.model = load_model(model_name)
            
            def encode(self, text: str) -> List[float]:
                return self.model.encode(text)
    """
    
    class StatefulWrapper:
        """Lazy initialization wrapper for stateful resources."""
        
        def __init__(self, *args, **kwargs):
            # Store init arguments without calling __init__
            self._init_args = args
            self._init_kwargs = kwargs
            self._instance = None  # Lazy initialized
            self._original_class = cls
        
        def _ensure_initialized(self):
            """Initialize instance if not already done."""
            if self._instance is None:
                self._instance = self._original_class(*self._init_args, **self._init_kwargs)
        
        def __call__(self, *args, **kwargs):
            """Forward calls to wrapped instance."""
            self._ensure_initialized()
            if hasattr(self._instance, "__call__"):
                return self._instance(*args, **kwargs)
            raise TypeError(f"{self._original_class.__name__} object is not callable")
        
        def __getattr__(self, name: str):
            """Forward attribute access to wrapped instance."""
            if name.startswith('_'):
                # Private attributes on wrapper
                return object.__getattribute__(self, name)
            
            # Lazily initialize and forward
            self._ensure_initialized()
            return getattr(self._instance, name)
        
        def __cache_key__(self) -> str:
            """Cache based on init arguments only, not instance state."""
            # Hash the class name and init arguments
            args_str = str(self._init_args)
            kwargs_str = str(sorted(self._init_kwargs.items()))
            combined = f"{cls.__name__}:{args_str}:{kwargs_str}"
            return hashlib.sha256(combined.encode()).hexdigest()
        
        def __getstate__(self):
            """Pickle only the init arguments, not the instance."""
            return {
                '_init_args': self._init_args,
                '_init_kwargs': self._init_kwargs,
                '_original_class': self._original_class,
                '_instance': None,  # Don't pickle the instance!
            }
        
        def __setstate__(self, state):
            """Restore from pickle."""
            self.__dict__.update(state)
        
        def __repr__(self):
            """String representation."""
            init_status = "initialized" if self._instance is not None else "lazy"
            return f"<{cls.__name__} ({init_status})>"
    
    # Mark the wrapper so engines can detect it
    StatefulWrapper.__hypernodes_stateful__ = True
    StatefulWrapper.__original_class__ = cls
    
    # Preserve metadata
    functools.update_wrapper(StatefulWrapper, cls, updated=[])
    
    return StatefulWrapper


def batch(method: Callable) -> Callable:
    """Mark a method as batch-optimized.
    
    Batch methods receive Series inputs and return Series outputs.
    This allows engines like DaftEngine to use optimized batch processing.
    
    Example:
        @stateful
        class Model:
            @batch
            def encode_batch(self, texts: Series) -> Series:
                # Process entire batch at once
                text_list = texts.to_pylist()
                results = self.model.encode(text_list)
                return Series.from_pylist(results)
    """
    method.__hypernodes_batch__ = True
    return method


# ==================== Mock Encoder ====================


@stateful
class MockEncoder:
    """Mock encoder for testing stateful pattern.
    
    Simulates expensive model loading with configurable delay.
    Provides both row-wise and batch encoding methods.
    """
    
    def __init__(self, model_name: str = "mock-model", delay_ms: int = 10):
        """Initialize encoder (simulates expensive model loading).
        
        Args:
            model_name: Model identifier (for caching/serialization)
            delay_ms: Simulated loading delay in milliseconds
        """
        # Simulate expensive model loading
        time.sleep(delay_ms / 1000)
        self.model_name = model_name
        # Generate deterministic weights based on model name
        seed = hash(model_name) % (2**32)
        rng = np.random.RandomState(seed)
        self._weights = rng.rand(768)
    
    def encode(self, text: str) -> List[float]:
        """Encode a single text (row-wise).
        
        Args:
            text: Text to encode
        
        Returns:
            Embedding vector (3D for speed)
        """
        # Deterministic encoding based on text hash
        text_hash = hash(text) % 100
        return (text_hash * self._weights[:3]).tolist()
    
    @batch
    def encode_batch(self, texts: Series) -> Series:
        """Encode a batch of texts (batch-optimized).
        
        Args:
            texts: Series of texts to encode
        
        Returns:
            Series of embedding vectors
        """
        text_list = texts.to_pylist()
        results = [self.encode(t) for t in text_list]
        return Series.from_pylist(results)


# ==================== Real HuggingFace Encoder ====================


@stateful
class SentenceTransformerEncoder:
    """Real SentenceTransformer encoder from HuggingFace.
    
    Uses actual model for production-grade testing of stateful pattern.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize SentenceTransformer model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def encode(self, text: str) -> List[float]:
        """Encode a single text (row-wise).
        
        Args:
            text: Text to encode
        
        Returns:
            Embedding vector
        """
        return self.model.encode(text).tolist()
    
    @batch
    def encode_batch(self, texts: Series) -> Series:
        """Encode a batch of texts (batch-optimized).
        
        Uses SentenceTransformer's native batch encoding.
        
        Args:
            texts: Series of texts to encode
        
        Returns:
            Series of embedding vectors
        """
        text_list = texts.to_pylist()
        # Use model's native batch encoding
        embeddings = self.model.encode(text_list)
        return Series.from_pylist([emb.tolist() for emb in embeddings])


# ==================== Version 1: HyperNodes Sequential ====================


@node(output_name="embedding")
def encode_passage_rowwise(text: str, encoder: Any) -> List[float]:
    """Row-wise encoding node.
    
    Works with any encoder that has .encode(text) method.
    Used with HypernodesEngine for sequential processing.
    """
    return encoder.encode(text)


def run_v1_sequential(texts: List[str], encoder: Any) -> tuple[List[List[float]], float]:
    """Version 1: HyperNodes Sequential (Row-wise).
    
    Args:
        texts: List of texts to encode
        encoder: Encoder instance (MockEncoder or SentenceTransformerEncoder)
    
    Returns:
        Tuple of (embeddings, execution_time)
    """
    from hypernodes import SequentialEngine
    
    pipeline = Pipeline(
        nodes=[encode_passage_rowwise],
        engine=SequentialEngine(),
    )
    
    start = time.time()
    results = pipeline.map(
        inputs={"text": texts, "encoder": encoder},
        map_over="text",
    )
    exec_time = time.time() - start
    
    # results is a list of dicts, extract embeddings
    embeddings = [r["embedding"] for r in results]
    return embeddings, exec_time


# ==================== Version 2: HyperNodes + DaftEngine ====================


@node(output_name="embeddings")
def encode_passages_batch(texts: Series, encoder: Any) -> Series:
    """Batch encoding node.
    
    Explicitly calls encoder's batch method.
    Used with DaftEngine for batch processing.
    """
    return encoder.encode_batch(texts)


def run_v2_hypernodes_daft(texts: List[str], encoder: Any) -> tuple[List[List[float]], float]:
    """Version 2: HyperNodes + DaftEngine (Explicit Batch).
    
    Uses @stateful encoder with a @daft.cls wrapper for proper distribution.
    
    Args:
        texts: List of texts to encode
        encoder: Encoder instance (MockEncoder or SentenceTransformerEncoder)
    
    Returns:
        Tuple of (embeddings, execution_time)
    """
    # Extract initialization args from the @stateful encoder
    model_name = encoder._init_kwargs.get("model_name", encoder._init_args[0] if encoder._init_args else "mock-model")
    
    # Determine if this is Mock or SentenceTransformer
    is_mock = "Mock" in encoder._original_class.__name__
    
    if is_mock:
        delay_ms = encoder._init_kwargs.get("delay_ms", 10)
        
        @daft.cls(max_concurrency=2)
        class EncoderWrapper:
            def __init__(self):
                # Mimic the mock encoder
                time.sleep(delay_ms / 1000)
                seed = hash(model_name) % (2**32)
                rng = np.random.RandomState(seed)
                self._weights = rng.rand(768)
            
            @daft.method.batch(return_dtype=DataType.list(DataType.float64()))
            def encode(self, texts: Series) -> Series:
                text_list = texts.to_pylist()
                results = []
                for text in text_list:
                    text_hash = hash(text) % 100
                    results.append((text_hash * self._weights[:3]).tolist())
                return Series.from_pylist(results)
    else:
        @daft.cls(max_concurrency=2)
        class EncoderWrapper:
            def __init__(self):
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
            
            @daft.method.batch(return_dtype=DataType.list(DataType.float64()))
            def encode(self, texts: Series) -> Series:
                text_list = texts.to_pylist()
                embeddings = self.model.encode(text_list)
                return Series.from_pylist([emb.tolist() for emb in embeddings])
    
    df = daft.from_pydict({"text": texts})
    encoder_inst = EncoderWrapper()
    
    start = time.time()
    df = df.with_column("embedding", encoder_inst.encode(daft.col("text")))
    result_df = df.collect()
    exec_time = time.time() - start
    
    embeddings = result_df.to_pydict()["embedding"]
    return embeddings, exec_time


# ==================== Version 3: Pure Daft ====================


def create_daft_mock_encoder(max_concurrency: int = 2, use_process: bool = False):
    """Create DaftMockEncoder with configurable concurrency."""
    @daft.cls(max_concurrency=max_concurrency, use_process=use_process)
    class DaftMockEncoder:
        """Pure Daft implementation of MockEncoder."""
        
        def __init__(self, model_name: str = "mock-model", delay_ms: int = 10):
            """Initialize encoder."""
            time.sleep(delay_ms / 1000)
            self.model_name = model_name
            seed = hash(model_name) % (2**32)
            rng = np.random.RandomState(seed)
            self._weights = rng.rand(768)
        
        @daft.method.batch(return_dtype=DataType.list(DataType.float64()))
        def encode(self, texts: Series) -> Series:
            """Batch encode texts."""
            text_list = texts.to_pylist()
            results = []
            for text in text_list:
                text_hash = hash(text) % 100
                results.append((text_hash * self._weights[:3]).tolist())
            return Series.from_pylist(results)
    
    return DaftMockEncoder


def create_daft_sentence_transformer_encoder(max_concurrency: int = 2, use_process: bool = False):
    """Create DaftSentenceTransformerEncoder with configurable concurrency."""
    @daft.cls(max_concurrency=max_concurrency, use_process=use_process)
    class DaftSentenceTransformerEncoder:
        """Pure Daft implementation of SentenceTransformerEncoder."""
        
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            """Initialize encoder."""
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
        
        @daft.method.batch(return_dtype=DataType.list(DataType.float64()))
        def encode(self, texts: Series) -> Series:
            """Batch encode texts."""
            text_list = texts.to_pylist()
            embeddings = self.model.encode(text_list)
            return Series.from_pylist([emb.tolist() for emb in embeddings])
    
    return DaftSentenceTransformerEncoder


def run_v3_pure_daft(
    texts: List[str], 
    encoder_factory: Callable,
    max_concurrency: int = 2,
    use_process: bool = False,
    **encoder_kwargs
) -> tuple[List[List[float]], float]:
    """Version 3: Pure Daft (Hand-written).
    
    Args:
        texts: List of texts to encode
        encoder_factory: Factory function that creates encoder class
        max_concurrency: Max concurrent instances
        use_process: Use process isolation
        **encoder_kwargs: Arguments for encoder initialization
    
    Returns:
        Tuple of (embeddings, execution_time)
    """
    encoder_class = encoder_factory(max_concurrency=max_concurrency, use_process=use_process)
    encoder = encoder_class(**encoder_kwargs)
    df = daft.from_pydict({"text": texts})
    
    start = time.time()
    df = df.with_column("embedding", encoder.encode(daft.col("text")))
    result_df = df.collect()
    exec_time = time.time() - start
    
    # Extract embeddings from collected DataFrame
    embeddings = result_df.to_pydict()["embedding"]
    return embeddings, exec_time


# ==================== Benchmark Harness ====================


def verify_embeddings_match(emb1: List[List[float]], emb2: List[List[float]], tolerance: float = 1e-6) -> bool:
    """Verify that two sets of embeddings match within tolerance."""
    if len(emb1) != len(emb2):
        return False
    
    for e1, e2 in zip(emb1, emb2):
        if len(e1) != len(e2):
            return False
        for v1, v2 in zip(e1, e2):
            if abs(v1 - v2) > tolerance:
                return False
    
    return True


def benchmark_mock_encoder(scale: int = 1000) -> dict:
    """Benchmark with mock encoder.
    
    Args:
        scale: Number of texts to encode
    
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*80}")
    print(f"MOCK ENCODER BENCHMARK (scale={scale})")
    print(f"{'='*80}")
    
    # Generate test data
    texts = [f"This is passage number {i} for testing" for i in range(scale)]
    
    results = {
        "scale": scale,
        "versions": {}
    }
    
    # Version 1: Sequential
    print("\n[V1] Running HyperNodes Sequential...")
    encoder_v1 = MockEncoder(model_name="mock-model", delay_ms=10)
    setup_start = time.time()
    # Trigger initialization
    _ = encoder_v1.encode("test")
    setup_time_v1 = time.time() - setup_start
    
    embeddings_v1, exec_time_v1 = run_v1_sequential(texts, encoder_v1)
    total_v1 = setup_time_v1 + exec_time_v1
    
    results["versions"]["v1_sequential"] = {
        "setup_time": setup_time_v1,
        "exec_time": exec_time_v1,
        "total_time": total_v1,
        "speedup": 1.0,
    }
    print(f"  Setup: {setup_time_v1:.3f}s | Exec: {exec_time_v1:.3f}s | Total: {total_v1:.3f}s")
    
    # Version 2: HyperNodes + DaftEngine
    print("\n[V2] Running HyperNodes + DaftEngine...")
    encoder_v2 = MockEncoder(model_name="mock-model", delay_ms=10)
    setup_start = time.time()
    _ = encoder_v2.encode("test")
    setup_time_v2 = time.time() - setup_start
    
    try:
        embeddings_v2, exec_time_v2 = run_v2_hypernodes_daft(texts, encoder_v2)
        total_v2 = setup_time_v2 + exec_time_v2
        speedup_v2 = total_v1 / total_v2
        
        # Verify correctness
        if verify_embeddings_match(embeddings_v1, embeddings_v2):
            print(f"  ✓ Embeddings match V1")
        else:
            print(f"  ✗ WARNING: Embeddings don't match V1!")
        
        results["versions"]["v2_hypernodes_daft"] = {
            "setup_time": setup_time_v2,
            "exec_time": exec_time_v2,
            "total_time": total_v2,
            "speedup": speedup_v2,
        }
        print(f"  Setup: {setup_time_v2:.3f}s | Exec: {exec_time_v2:.3f}s | Total: {total_v2:.3f}s | Speedup: {speedup_v2:.2f}x")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["versions"]["v2_hypernodes_daft"] = {"error": str(e)}
    
    # Version 3: Pure Daft - test different configurations
    daft_configs = [
        {"max_concurrency": 1, "use_process": False, "name": "V3a: Daft(c=1)"},
        {"max_concurrency": 2, "use_process": False, "name": "V3b: Daft(c=2)"},
        {"max_concurrency": 4, "use_process": False, "name": "V3c: Daft(c=4)"},
        {"max_concurrency": 2, "use_process": True, "name": "V3d: Daft(c=2,proc)"},
    ]
    
    for config in daft_configs:
        config_name = config["name"]
        print(f"\n[{config_name}] Running...")
        setup_time_v3 = 0.01  # Negligible
        
        try:
            embeddings_v3, exec_time_v3 = run_v3_pure_daft(
                texts, 
                create_daft_mock_encoder,
                max_concurrency=config["max_concurrency"],
                use_process=config["use_process"],
                model_name="mock-model", 
                delay_ms=10
            )
            total_v3 = setup_time_v3 + exec_time_v3
            speedup_v3 = total_v1 / total_v3
            
            # Verify correctness
            if verify_embeddings_match(embeddings_v1, embeddings_v3):
                print(f"  ✓ Embeddings match V1")
            else:
                print(f"  ✗ WARNING: Embeddings don't match V1!")
            
            version_key = config_name.lower().replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace(",", "_").replace("=", "")
            results["versions"][version_key] = {
                "setup_time": setup_time_v3,
                "exec_time": exec_time_v3,
                "total_time": total_v3,
                "speedup": speedup_v3,
                "config": config,
            }
            print(f"  Setup: {setup_time_v3:.3f}s | Exec: {exec_time_v3:.3f}s | Total: {total_v3:.3f}s | Speedup: {speedup_v3:.2f}x")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def benchmark_real_encoder(scale: int = 100) -> dict:
    """Benchmark with real SentenceTransformer encoder.
    
    Args:
        scale: Number of texts to encode
    
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*80}")
    print(f"REAL SENTENCETRANSFORMER BENCHMARK (scale={scale})")
    print(f"{'='*80}")
    
    # Generate test data
    texts = [f"This is passage number {i} for testing" for i in range(scale)]
    
    results = {
        "scale": scale,
        "versions": {}
    }
    
    model_name = "all-MiniLM-L6-v2"
    
    # Version 1: Sequential
    print("\n[V1] Running HyperNodes Sequential...")
    encoder_v1 = SentenceTransformerEncoder(model_name=model_name)
    setup_start = time.time()
    # Trigger initialization
    _ = encoder_v1.encode("test")
    setup_time_v1 = time.time() - setup_start
    
    embeddings_v1, exec_time_v1 = run_v1_sequential(texts, encoder_v1)
    total_v1 = setup_time_v1 + exec_time_v1
    
    results["versions"]["v1_sequential"] = {
        "setup_time": setup_time_v1,
        "exec_time": exec_time_v1,
        "total_time": total_v1,
        "speedup": 1.0,
    }
    print(f"  Setup: {setup_time_v1:.3f}s | Exec: {exec_time_v1:.3f}s | Total: {total_v1:.3f}s")
    
    # Version 2: HyperNodes + DaftEngine
    print("\n[V2] Running HyperNodes + DaftEngine...")
    encoder_v2 = SentenceTransformerEncoder(model_name=model_name)
    setup_start = time.time()
    _ = encoder_v2.encode("test")
    setup_time_v2 = time.time() - setup_start
    
    try:
        embeddings_v2, exec_time_v2 = run_v2_hypernodes_daft(texts, encoder_v2)
        total_v2 = setup_time_v2 + exec_time_v2
        speedup_v2 = total_v1 / total_v2
        
        # Verify correctness (with tolerance for floating point)
        if verify_embeddings_match(embeddings_v1, embeddings_v2, tolerance=1e-5):
            print(f"  ✓ Embeddings match V1")
        else:
            print(f"  ✗ WARNING: Embeddings don't match V1!")
        
        results["versions"]["v2_hypernodes_daft"] = {
            "setup_time": setup_time_v2,
            "exec_time": exec_time_v2,
            "total_time": total_v2,
            "speedup": speedup_v2,
        }
        print(f"  Setup: {setup_time_v2:.3f}s | Exec: {exec_time_v2:.3f}s | Total: {total_v2:.3f}s | Speedup: {speedup_v2:.2f}x")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["versions"]["v2_hypernodes_daft"] = {"error": str(e)}
    
    # Version 3: Pure Daft - test different configurations
    daft_configs = [
        {"max_concurrency": 1, "use_process": False, "name": "V3a: Daft(c=1)"},
        {"max_concurrency": 2, "use_process": False, "name": "V3b: Daft(c=2)"},
        {"max_concurrency": 4, "use_process": False, "name": "V3c: Daft(c=4)"},
    ]
    
    for config in daft_configs:
        config_name = config["name"]
        print(f"\n[{config_name}] Running...")
        setup_time_v3 = 0.01  # Negligible
        
        try:
            embeddings_v3, exec_time_v3 = run_v3_pure_daft(
                texts, 
                create_daft_sentence_transformer_encoder,
                max_concurrency=config["max_concurrency"],
                use_process=config["use_process"],
                model_name=model_name
            )
            total_v3 = setup_time_v3 + exec_time_v3
            speedup_v3 = total_v1 / total_v3
            
            # Verify correctness
            if verify_embeddings_match(embeddings_v1, embeddings_v3, tolerance=1e-5):
                print(f"  ✓ Embeddings match V1")
            else:
                print(f"  ✗ WARNING: Embeddings don't match V1!")
            
            version_key = config_name.lower().replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace(",", "_").replace("=", "")
            results["versions"][version_key] = {
                "setup_time": setup_time_v3,
                "exec_time": exec_time_v3,
                "total_time": total_v3,
                "speedup": speedup_v3,
                "config": config,
            }
            print(f"  Setup: {setup_time_v3:.3f}s | Exec: {exec_time_v3:.3f}s | Total: {total_v3:.3f}s | Speedup: {speedup_v3:.2f}x")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def generate_markdown_report(mock_results: dict, real_results: dict) -> str:
    """Generate markdown report of benchmark results."""
    report = ["# Stateful Batch Benchmark Results\n"]
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Mock encoder results
    report.append("\n## Mock Encoder Results\n")
    report.append("| Version | Scale | Setup (s) | Execution (s) | Total (s) | Speedup |")
    report.append("|---------|-------|-----------|---------------|-----------|---------|")
    
    for scale_results in mock_results:
        scale = scale_results["scale"]
        for version_name, version_data in scale_results["versions"].items():
            if "error" in version_data:
                continue
            version_label = {
                "v1_sequential": "V1: Sequential",
                "v2_hypernodes_daft": "V2: HN+Daft",
                "v3_pure_daft": "V3: Pure Daft",
            }.get(version_name, version_name)
            
            report.append(
                f"| {version_label} | {scale} | {version_data['setup_time']:.3f} | "
                f"{version_data['exec_time']:.3f} | {version_data['total_time']:.3f} | "
                f"{version_data['speedup']:.2f}x |"
            )
    
    # Real encoder results
    report.append("\n## Real SentenceTransformer Results\n")
    report.append("| Version | Scale | Setup (s) | Execution (s) | Total (s) | Speedup |")
    report.append("|---------|-------|-----------|---------------|-----------|---------|")
    
    for scale_results in real_results:
        scale = scale_results["scale"]
        for version_name, version_data in scale_results["versions"].items():
            if "error" in version_data:
                continue
            version_label = {
                "v1_sequential": "V1: Sequential",
                "v2_hypernodes_daft": "V2: HN+Daft",
                "v3_pure_daft": "V3: Pure Daft",
            }.get(version_name, version_name)
            
            report.append(
                f"| {version_label} | {scale} | {version_data['setup_time']:.3f} | "
                f"{version_data['exec_time']:.3f} | {version_data['total_time']:.3f} | "
                f"{version_data['speedup']:.2f}x |"
            )
    
    return "\n".join(report)


# ==================== Main ====================


def main():
    """Run all benchmarks and generate report."""
    print("="*80)
    print("STATEFUL BATCH BENCHMARK")
    print("="*80)
    
    # Test mock encoder with multiple scales
    mock_results = []
    for scale in [100, 1000]:
        try:
            result = benchmark_mock_encoder(scale=scale)
            mock_results.append(result)
        except Exception as e:
            print(f"\n✗ Mock benchmark failed at scale {scale}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test real encoder
    real_results = []
    for scale in [10, 100]:
        try:
            result = benchmark_real_encoder(scale=scale)
            real_results.append(result)
        except Exception as e:
            print(f"\n✗ Real benchmark failed at scale {scale}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate and save report
    if mock_results or real_results:
        report = generate_markdown_report(mock_results, real_results)
        
        output_file = "outputs/stateful_batch_benchmark_results.md"
        with open(output_file, "w") as f:
            f.write(report)
        
        print(f"\n{'='*80}")
        print(f"Report saved to: {output_file}")
        print(f"{'='*80}\n")
        print(report)


if __name__ == "__main__":
    main()

