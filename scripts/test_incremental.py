#!/usr/bin/env python3
"""
Incremental test - start tiny and build up to find safe parameters.
"""

import time
from sentence_transformers import CrossEncoder
import torch

print("=" * 60)
print("INCREMENTAL CROSS-ENCODER TEST")
print("=" * 60)

# Step 1: Test tiny batch
print("\nStep 1: Loading model...")
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print(f"✓ Model loaded")
print(f"  Device: {model._target_device if hasattr(model, '_target_device') else 'unknown'}")

# Step 2: Test 1 pair
print("\nStep 2: Testing 1 pair...")
pairs = [["test query", "test document"]]
start = time.perf_counter()
scores = model.predict(pairs, batch_size=1, show_progress_bar=False)
elapsed = time.perf_counter() - start
print(f"✓ 1 pair: {elapsed:.3f}s, score={scores[0]:.2f}")

# Step 3: Test 10 pairs
print("\nStep 3: Testing 10 pairs...")
pairs = [["query", "document"]] * 10
start = time.perf_counter()
scores = model.predict(pairs, batch_size=10, show_progress_bar=False)
elapsed = time.perf_counter() - start
print(f"✓ 10 pairs: {elapsed:.3f}s, throughput={10/elapsed:.1f} pairs/sec")

# Step 4: Test 100 pairs
print("\nStep 4: Testing 100 pairs (batch_size=32)...")
pairs = [["query", "document"]] * 100
start = time.perf_counter()
scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
elapsed = time.perf_counter() - start
print(f"✓ 100 pairs: {elapsed:.3f}s, throughput={100/elapsed:.1f} pairs/sec")

# Step 5: Test 300 pairs (the actual benchmark size)
print("\nStep 5: Testing 300 pairs (batch_size=32)...")
pairs = [["query", "document"]] * 300
start = time.perf_counter()
scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
elapsed = time.perf_counter() - start
print(f"✓ 300 pairs: {elapsed:.3f}s, throughput={300/elapsed:.1f} pairs/sec")

# Step 6: Test larger batch size
print("\nStep 6: Testing 300 pairs (batch_size=64)...")
start = time.perf_counter()
scores = model.predict(pairs, batch_size=64, show_progress_bar=False)
elapsed = time.perf_counter() - start
print(f"✓ 300 pairs (batch=64): {elapsed:.3f}s, throughput={300/elapsed:.1f} pairs/sec")

# Step 7: Test even larger batch size
print("\nStep 7: Testing 300 pairs (batch_size=128)...")
start = time.perf_counter()
scores = model.predict(pairs, batch_size=128, show_progress_bar=False)
elapsed = time.perf_counter() - start
print(f"✓ 300 pairs (batch=128): {elapsed:.3f}s, throughput={300/elapsed:.1f} pairs/sec")

print("\n" + "=" * 60)
print("✓ All incremental tests passed!")
print("=" * 60)


