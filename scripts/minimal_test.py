#!/usr/bin/env python3
from sentence_transformers import CrossEncoder
import torch

print("Test 1: Basic model")
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [['What is machine learning?', 'Machine learning is AI']]
scores = model.predict(pairs)
print('Scores (no inference mode):', scores)

print("\nTest 2: With inference mode")
with torch.inference_mode():
    scores2 = model.predict(pairs)
print('Scores (with inference mode):', scores2)

print("\nTest 3: Re-using model for second prediction")
scores3 = model.predict(pairs)
print('Scores (second call):', scores3)

print("\nTest 4: With warmup pattern (like my class)")
# Do a warmup prediction
warmup_pairs = [["warmup query", "warmup document"]] * 32
with torch.inference_mode():
    model.predict(warmup_pairs, batch_size=32, show_progress_bar=False)
    
# Now predict again
with torch.inference_mode():
    scores4 = model.predict(pairs, batch_size=32, show_progress_bar=False)
print('Scores (after warmup):', scores4)

