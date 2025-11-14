#!/usr/bin/env python3
"""Debug cross-encoder to find NaN issue."""

import torch
from sentence_transformers import CrossEncoder

print("Testing raw CrossEncoder...")
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [
    ["What is machine learning?", "Machine learning is a subset of AI."],
    ["How does photosynthesis work?", "Plants convert light to energy."],
]

print("\nTest 1: Basic predict")
scores = model.predict(pairs)
print(f"Scores: {scores}")
print(f"Type: {type(scores)}")

print("\nTest 2: With inference_mode")
with torch.inference_mode():
    scores2 = model.predict(pairs)
    print(f"Scores: {scores2}")

print("\nTest 3: Single pair")
single_score = model.predict([["query", "document"]])
print(f"Single score: {single_score}")



