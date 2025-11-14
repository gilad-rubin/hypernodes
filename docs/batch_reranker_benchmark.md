# Batched CrossEncoder Reranking Benchmark

## Context
- Pipeline: `scripts/retrieval_ultra_fast.py`
- Engine: `SequentialEngine`
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (MPS backend on Apple Silicon)
- Dataset variants: `data/sample_5` and `data/sample_10` (`dev` split)
- Command template:
  ```bash
  uv run scripts/retrieval_ultra_fast.py \
    --engine sequential \
    --examples <N> \
    --split dev \
    --reranker-batch-size <batch> \
    --quiet
  ```
- `reranker_batch_size=0` preserves the legacy per-query reranking loop.

## Results

| Dataset | `reranker_batch_size` | `batch_rerank_predictions` runtime | Total runtime | Notes |
|---------|----------------------|------------------------------------|---------------|-------|
| sample_5 | 0 (legacy) | 13.94 s | 17.23 s | CrossEncoder called once per query |
| sample_5 | 128 (batched) | 14.29 s | 17.46 s | Single batched predict (~300 pairs) |
| sample_10 | 0 (legacy) | 27.36 s | 32.63 s | Input pairs ≈ 600 |
| sample_10 | 128 (batched) | 29.43 s | 34.65 s | Slight overhead from building pair list |
| sample_10 | 512 | ❌ OOM | — | MPS ran out of memory (~6 GiB request) |

## Observations
- The new `batch_rerank_predictions` node works end-to-end: setting `reranker_batch_size > 0` aggregates all `(query, passage)` pairs, runs `CrossEncoder.predict()` once, and rehydrates per-query predictions.
- On this Apple Silicon machine the batched path is comparable/slightly slower because:
  - The MPS backend already executes multiple smaller `predict` calls efficiently.
  - Building the global pair list and mapping scores back per query adds overhead for small datasets.
  - Aggressive batching (≥512) exhausts the MPS memory pool; CUDA hardware should benefit more.
- Users can toggle the behavior via the new CLI flag without touching pipeline code, enabling experimentation on larger GPUs.

## Next Steps
1. Re-run on CUDA hardware (or with `torch.compile` enabled) to validate batching benefits at >5k pairs.
2. If batching proves consistently faster, set a higher default `reranker_batch_size` or auto-tune it based on dataset size.
3. Consider engine-level batching support so any map-heavy node can opt into similar treatment.

