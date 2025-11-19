import os
from typing import Any, List, Protocol, Dict, Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel
from hypernodes import Pipeline, node

# Ensure assets directory exists
ASSETS_DIR = "docs/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# =============================================================================
# 1. Real Pipeline Code (from notebooks/retrieval.ipynb)
# =============================================================================

# --- Pydantic Models ---
class Passage(BaseModel):
    uuid: str
    text: str
    model_config = {"frozen": True}

class EncodedPassage(BaseModel):
    uuid: str
    text: str
    embedding: Any
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

class Query(BaseModel):
    uuid: str
    text: str
    model_config = {"frozen": True}

class EncodedQuery(BaseModel):
    uuid: str
    text: str
    embedding: Any
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

class SearchHit(BaseModel):
    passage_uuid: str
    score: float
    model_config = {"frozen": True}

class Prediction(BaseModel):
    query_uuid: str
    paragraph_uuid: str
    score: float
    model_config = {"frozen": True}

class GroundTruth(BaseModel):
    query_uuid: str
    paragraph_uuid: str
    label_score: int
    model_config = {"frozen": True}

# --- Protocols ---
class Encoder(Protocol):
    def encode(self, text: str, is_query: bool = False) -> Any: ...

class VectorIndex(Protocol):
    def search(self, query_embedding: Any, k: int) -> List[SearchHit]: ...

class BM25Index(Protocol):
    def search(self, query_text: str, k: int) -> List[SearchHit]: ...

class Reranker(Protocol):
    def rerank(self, query: Query, candidates: List[SearchHit], k: int) -> List[SearchHit]: ...

# --- Implementation Classes (Mocked for Visualization) ---
class Model2VecEncoder:
    def __init__(self, model_name: str): pass
    def encode(self, text: str, is_query: bool = False) -> np.ndarray: return np.array([])

class CosineSimIndex:
    def __init__(self, encoded_passages: List[EncodedPassage]): pass
    def search(self, query_embedding: np.ndarray, k: int) -> List[SearchHit]: return []

class BM25IndexImpl:
    def __init__(self, passages: List[Passage]): pass
    def search(self, query_text: str, k: int) -> List[SearchHit]: return []

class CrossEncoderReranker(BaseModel):
    model_name: str
    model_config = {"arbitrary_types_allowed": True, "frozen": True}
    def rerank(self, query: Query, candidates: List[SearchHit], k: int, encoded_passages: List[EncodedPassage]) -> List[SearchHit]: return []

class RRFFusion:
    def __init__(self, k: int = 60): pass
    def fuse(self, results_list: List[List[SearchHit]]) -> List[SearchHit]: return []

class NDCGEvaluator:
    def __init__(self, k: int): pass
    def compute(self, predictions: List[Prediction], ground_truths: List[GroundTruth]) -> float: return 0.0

class RecallEvaluator:
    def __init__(self, k_list: List[int]): pass
    def compute(self, predictions: List[Prediction], ground_truths: List[GroundTruth]) -> dict[str, float]: return {}

# --- Nodes ---
@node(output_name="passages")
def load_passages(corpus_path: str, limit: int = 0) -> List[Passage]:
    return []

@node(output_name="queries")
def load_queries(examples_path: str) -> List[Query]:
    return []

@node(output_name="ground_truths")
def load_ground_truths(examples_path: str) -> List[GroundTruth]:
    return []

@node(output_name="encoder")
def create_encoder(model_name: str) -> Encoder:
    return Model2VecEncoder(model_name)

@node(output_name="rrf")
def create_rrf_fusion(rrf_k: int = 60) -> RRFFusion:
    return RRFFusion(k=rrf_k)

@node(output_name="ndcg_evaluator")
def create_ndcg_evaluator(ndcg_k: int) -> NDCGEvaluator:
    return NDCGEvaluator(k=ndcg_k)

@node(output_name="recall_evaluator")
def create_recall_evaluator(recall_k_list: List[int]) -> RecallEvaluator:
    return RecallEvaluator(k_list=recall_k_list)

@node(output_name="vector_index")
def build_vector_index(encoded_passages: List[EncodedPassage]) -> VectorIndex:
    return CosineSimIndex(encoded_passages)

@node(output_name="bm25_index")
def build_bm25_index(passages: List[Passage]) -> BM25Index:
    return BM25IndexImpl(passages)

@node(output_name="reranker")
def create_reranker(reranker_model_name: str, encoded_passages: List[EncodedPassage]) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=reranker_model_name)

@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
    return EncodedPassage(uuid="1", text="text", embedding=np.array([]))

@node(output_name="encoded_query")
def encode_query(query: Query, encoder: Encoder) -> EncodedQuery:
    return EncodedQuery(uuid="1", text="text", embedding=np.array([]))

@node(output_name="query")
def extract_query(encoded_query: EncodedQuery) -> Query:
    return Query(uuid="1", text="text")

@node(output_name="vector_hits")
def retrieve_vector(encoded_query: EncodedQuery, vector_index: VectorIndex, top_k: int) -> List[SearchHit]:
    return []

@node(output_name="bm25_hits")
def retrieve_bm25(query: Query, bm25_index: BM25Index, top_k: int) -> List[SearchHit]:
    return []

@node(output_name="fused_hits")
def fuse_results(vector_hits: List[SearchHit], bm25_hits: List[SearchHit], rrf: RRFFusion) -> List[SearchHit]:
    return []

@node(output_name="reranked_hits")
def rerank_with_crossencoder(query: Query, fused_hits: List[SearchHit], reranker: CrossEncoderReranker, encoded_passages: List[EncodedPassage], rerank_k: int) -> List[SearchHit]:
    return []

@node(output_name="predictions")
def hits_to_predictions(query: Query, reranked_hits: List[SearchHit]) -> List[Prediction]:
    return []

@node(output_name="all_predictions")
def flatten_predictions(all_query_predictions: List[List[Prediction]]) -> List[Prediction]:
    return []

@node(output_name="ndcg_score")
def compute_ndcg(all_predictions: List[Prediction], ground_truths: List[GroundTruth], ndcg_evaluator: NDCGEvaluator) -> float:
    return 0.0

@node(output_name="recall_metrics")
def compute_recall(all_predictions: List[Prediction], ground_truths: List[GroundTruth], recall_evaluator: RecallEvaluator) -> dict[str, float]:
    return {}

@node(output_name="evaluation_results")
def combine_evaluation_results(ndcg_score: float, recall_metrics: dict[str, float], ndcg_k: int) -> dict:
    return {}

# --- Pipelines ---

# Single-item pipelines
encode_single_passage = Pipeline(nodes=[encode_passage], name="encode_single_passage")
encode_single_query = Pipeline(nodes=[encode_query], name="encode_single_query")
retrieve_single_query = Pipeline(
    nodes=[
        extract_query,
        retrieve_vector,
        retrieve_bm25,
        fuse_results,
        rerank_with_crossencoder,
        hits_to_predictions,
    ],
    name="retrieve_single_query",
)

# Mapped Nodes
encode_passages_mapped = encode_single_passage.as_node(
    input_mapping={"passages": "passage"},
    output_mapping={"encoded_passage": "encoded_passages"},
    map_over="passages",
    name="encode_passages_mapped",
)

encode_queries_mapped = encode_single_query.as_node(
    input_mapping={"queries": "query"},
    output_mapping={"encoded_query": "encoded_queries"},
    map_over="queries",
    name="encode_queries_mapped",
)

retrieve_queries_mapped = retrieve_single_query.as_node(
    input_mapping={"encoded_queries": "encoded_query"},
    output_mapping={"predictions": "all_query_predictions"},
    map_over="encoded_queries",
    name="retrieve_queries_mapped",
)

# Full Pipeline
pipeline = Pipeline(
    nodes=[
        load_passages,
        load_queries,
        load_ground_truths,
        create_encoder,
        create_rrf_fusion,
        create_ndcg_evaluator,
        create_recall_evaluator,
        encode_passages_mapped,
        build_vector_index,
        build_bm25_index,
        create_reranker,
        encode_queries_mapped,
        retrieve_queries_mapped,
        flatten_predictions,
        compute_ndcg,
        compute_recall,
        combine_evaluation_results,
    ],
    name="hebrew_retrieval",
)

# --- Generate Pipeline SVGs ---

# Simple intro example for introduction page
@node(output_name="cleaned_text")
def clean_text_intro(passage: str) -> str:
    return passage.strip().lower()

@node(output_name="word_count")
def count_words_intro(cleaned_text: str) -> int:
    return len(cleaned_text.split())

simple_pipeline = Pipeline(
    nodes=[clean_text_intro, count_words_intro],
    name="text_processor"
)
simple_pipeline.visualize(filename=os.path.join(ASSETS_DIR, "pipeline_simple.svg"))

# Full retrieval pipeline overview
pipeline.visualize(filename=os.path.join(ASSETS_DIR, "pipeline_overview.svg"))

# Nested Views
pipeline.visualize(filename=os.path.join(ASSETS_DIR, "pipeline_nested_collapsed.svg"), depth=1)
pipeline.visualize(filename=os.path.join(ASSETS_DIR, "pipeline_nested_expanded.svg"), depth=None)


# =============================================================================
# 2. Real Progress Bar Visualization (Rich Style)
# =============================================================================
# Mimicking Rich's progress bar style with SVG

rich_progress_svg = """<svg width="700" height="120" viewBox="0 0 700 120" xmlns="http://www.w3.org/2000/svg">
  <style>
    .bg { fill: #0F111A; }
    .text { font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 14px; fill: #A6ACCD; }
    .white { fill: #FFFFFF; }
    .blue { fill: #82AAFF; }
    .green { fill: #C3E88D; }
    .purple { fill: #C792EA; }
    .dim { fill: #444; }
    .bar-bg { fill: #292D3E; }
    .bar-fill { fill: #F07178; } /* Rich default red/pinkish */
    .bar-pulse { fill: #FFCB6B; }
  </style>
  
  <rect width="100%" height="100%" class="bg" rx="6" />
  
  <!-- Header -->
  <text x="20" y="30" class="text"><tspan class="purple" font-weight="bold">HyperNodes</tspan> Pipeline Execution</text>
  
  <!-- Main Pipeline Bar -->
  <text x="20" y="60" class="text">hebrew_retrieval</text>
  <rect x="160" y="50" width="300" height="12" class="bar-bg" rx="2" />
  <rect x="160" y="50" width="240" height="12" class="bar-fill" rx="2" /> <!-- 80% -->
  <text x="470" y="60" class="text purple">80%</text>
  <text x="510" y="60" class="text dim">00:12&lt;00:03</text>
  
  <!-- Sub-task Bar -->
  <text x="40" y="85" class="text">└─ retrieve_queries</text>
  <rect x="160" y="75" width="300" height="12" class="bar-bg" rx="2" />
  <rect x="160" y="75" width="120" height="12" class="bar-pulse" rx="2" /> <!-- 40% -->
  <text x="470" y="85" class="text blue">40%</text>
  <text x="510" y="85" class="text dim">00:05&lt;00:07</text>

  <!-- Footer -->
  <text x="20" y="108" class="text green">✓ Cached: build_vector_index, build_bm25_index</text>
</svg>"""

with open(os.path.join(ASSETS_DIR, "progress_bar.svg"), "w") as f:
    f.write(rich_progress_svg)


# =============================================================================
# 3. Real Telemetry Visualization (Jaeger/Honeycomb Style)
# =============================================================================

telemetry_svg = """<svg width="800" height="250" viewBox="0 0 800 250" xmlns="http://www.w3.org/2000/svg">
  <style>
    .bg { fill: #1e1e1e; }
    .text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #ccc; }
    .title { font-size: 14px; font-weight: bold; fill: #fff; }
    .axis { stroke: #444; stroke-width: 1; stroke-dasharray: 4; }
    .span-rect { rx: 3; stroke: rgba(0,0,0,0.2); stroke-width: 1; }
    .span-label { font-size: 11px; fill: #fff; pointer-events: none; }
    
    /* Colors */
    .c-pipeline { fill: #3B82F6; } /* Blue */
    .c-setup { fill: #8B5CF6; } /* Purple */
    .c-encode { fill: #EC4899; } /* Pink */
    .c-index { fill: #10B981; } /* Green */
    .c-retrieve { fill: #F59E0B; } /* Amber */
    .c-eval { fill: #6366F1; } /* Indigo */
  </style>
  
  <rect width="100%" height="100%" class="bg" rx="6" />
  
  <!-- Header -->
  <text x="20" y="25" class="title">Trace: hebrew_retrieval (Total Duration: 12.4s)</text>
  
  <!-- Time Axis -->
  <line x1="150" y1="40" x2="150" y2="230" class="axis" />
  <text x="155" y="245" class="text">2s</text>
  <line x1="300" y1="40" x2="300" y2="230" class="axis" />
  <text x="305" y="245" class="text">5s</text>
  <line x1="450" y1="40" x2="450" y2="230" class="axis" />
  <text x="455" y="245" class="text">8s</text>
  <line x1="600" y1="40" x2="600" y2="230" class="axis" />
  <text x="605" y="245" class="text">11s</text>

  <!-- Spans -->
  <!-- 1. Root Pipeline -->
  <rect x="20" y="50" width="760" height="20" class="span-rect c-pipeline" />
  <text x="25" y="64" class="span-label">hebrew_retrieval</text>

  <!-- 2. Setup (Load & Create) -->
  <rect x="20" y="75" width="80" height="18" class="span-rect c-setup" />
  <text x="25" y="88" class="span-label">setup</text>

  <!-- 3. Encode Passages (Parallel) -->
  <rect x="110" y="75" width="120" height="18" class="span-rect c-encode" />
  <text x="115" y="88" class="span-label">encode_passages (map)</text>

  <!-- 4. Build Indexes (Cached) -->
  <rect x="240" y="75" width="10" height="18" class="span-rect c-index" opacity="0.5" />
  <text x="255" y="88" class="text" fill="#10B981">build_index (cached)</text>

  <!-- 5. Encode Queries -->
  <rect x="260" y="75" width="40" height="18" class="span-rect c-encode" />
  <text x="265" y="88" class="span-label">encode_queries</text>

  <!-- 6. Retrieve (Longest) -->
  <rect x="310" y="75" width="400" height="18" class="span-rect c-retrieve" />
  <text x="315" y="88" class="span-label">retrieve_queries_mapped</text>
  
    <!-- 6.1 Retrieve Detail (Nested) -->
    <rect x="310" y="98" width="380" height="16" class="span-rect c-retrieve" opacity="0.8" />
    <text x="315" y="110" class="span-label">retrieve_vector</text>
    
    <rect x="330" y="118" width="340" height="16" class="span-rect c-retrieve" opacity="0.8" />
    <text x="335" y="130" class="span-label">rerank_crossencoder</text>

  <!-- 7. Evaluate -->
  <rect x="720" y="75" width="60" height="18" class="span-rect c-eval" />
  <text x="725" y="88" class="span-label">evaluate</text>

</svg>"""

with open(os.path.join(ASSETS_DIR, "trace_waterfall.svg"), "w") as f:
    f.write(telemetry_svg)

print("Real visualizations generated successfully!")
