import json
from hypernodes.viz.visualization_widget import generate_widget_html, transform_to_react_flow

# User provided JSON
graph_data = {
  "levels": [
    {
      "level_id": "root",
      "parent_level_id": None,
      "unfulfilled_inputs": ["eval_pairs"],
      "bound_inputs_at_this_level": [],
      "inherited_inputs": [],
      "local_output_mapping": {},
      "local_input_mapping": {},
      "parent_pipeline_node_id": None
    },
    {
      "level_id": "root__nested_4634062064",
      "parent_level_id": "root",
      "unfulfilled_inputs": ["eval_pair", "judge"],
      "bound_inputs_at_this_level": ["judge"],
      "inherited_inputs": [],
      "local_output_mapping": {},
      "local_input_mapping": {},
      "parent_pipeline_node_id": "4634062064"
    },
    {
      "level_id": "root__nested_4634062064__nested_4633684608",
      "parent_level_id": "root__nested_4634062064",
      "unfulfilled_inputs": ["llm", "query", "top_k", "vector_store"],
      "bound_inputs_at_this_level": ["vector_store", "llm", "top_k"],
      "inherited_inputs": [],
      "local_output_mapping": {},
      "local_input_mapping": {},
      "parent_pipeline_node_id": "4634062064__4633684608"
    }
  ],
  "nodes": [
    {
      "id": "4634062064",
      "level_id": "root",
      "node_type": "PIPELINE",
      "label": "batch_evaluation",
      "function_name": "batch_evaluation",
      "output_names": ["evaluation_results"],
      "inputs": [
        {
          "name": "eval_pairs",
          "type_hint": None,
          "default_value": None,
          "is_bound": False,
          "is_fulfilled_by_sibling": False
        }
      ],
      "is_expanded": True,
      "input_mapping": {"eval_pairs": "eval_pair"},
      "output_mapping": {"evaluation_result": "evaluation_results"}
    },
    {
      "id": "4634062064__4633915232",
      "level_id": "root__nested_4634062064",
      "node_type": "STANDARD",
      "label": "extract_query",
      "function_name": "extract_query",
      "output_names": ["query"],
      "inputs": [
        {
          "name": "eval_pair",
          "type_hint": "EvaluationPair",
          "default_value": None,
          "is_bound": False,
          "is_fulfilled_by_sibling": False
        }
      ]
    },
    {
      "id": "4634062064__4633684608",
      "level_id": "root__nested_4634062064",
      "node_type": "PIPELINE",
      "label": "retrieval",
      "function_name": "retrieval",
      "output_names": ["answer", "retrieved_docs"],
      "inputs": [
        {
          "name": "query",
          "type_hint": None,
          "default_value": None,
          "is_bound": False,
          "is_fulfilled_by_sibling": False
        }
      ],
      "is_expanded": True,
      "input_mapping": {},
      "output_mapping": {}
    },
    {
      "id": "4634062064__4633684608__4633688064",
      "level_id": "root__nested_4634062064__nested_4633684608",
      "node_type": "STANDARD",
      "label": "retrieve",
      "function_name": "retrieve",
      "output_names": ["retrieved_docs"],
      "inputs": [
        {
          "name": "query",
          "type_hint": "str",
          "default_value": None,
          "is_bound": False,
          "is_fulfilled_by_sibling": False
        },
        {
          "name": "vector_store",
          "type_hint": "VectorStore",
          "default_value": None,
          "is_bound": True,
          "is_fulfilled_by_sibling": False
        },
        {
          "name": "top_k",
          "type_hint": "int",
          "default_value": "2",
          "is_bound": True,
          "is_fulfilled_by_sibling": False
        }
      ]
    },
    {
      "id": "4634062064__4633684608__4633688016",
      "level_id": "root__nested_4634062064__nested_4633684608",
      "node_type": "STANDARD",
      "label": "generate",
      "function_name": "generate",
      "output_names": ["answer"],
      "inputs": [
        {
          "name": "query",
          "type_hint": "str",
          "default_value": None,
          "is_bound": False,
          "is_fulfilled_by_sibling": False
        },
        {
          "name": "retrieved_docs",
          "type_hint": "List[Document]",
          "default_value": None,
          "is_bound": False,
          "is_fulfilled_by_sibling": False
        },
        {
          "name": "llm",
          "type_hint": "LLM",
          "default_value": None,
          "is_bound": True,
          "is_fulfilled_by_sibling": False
        }
      ]
    },
    {
      "id": "4634062064__4633913024",
      "level_id": "root__nested_4634062064",
      "node_type": "STANDARD",
      "label": "evaluate_answer",
      "function_name": "evaluate_answer",
      "output_names": ["evaluation_result"],
      "inputs": [
        {
          "name": "eval_pair",
          "type_hint": "EvaluationPair",
          "default_value": None,
          "is_bound": False,
          "is_fulfilled_by_sibling": False
        },
        {
          "name": "answer",
          "type_hint": None,
          "default_value": None,
          "is_bound": False,
          "is_fulfilled_by_sibling": False
        },
        {
          "name": "judge",
          "type_hint": "JudgeEvaluator",
          "default_value": None,
          "is_bound": True,
          "is_fulfilled_by_sibling": False
        }
      ]
    },
    {
      "id": "4634243584",
      "level_id": "root",
      "node_type": "STANDARD",
      "label": "compute_metrics",
      "function_name": "compute_metrics",
      "output_names": ["metrics"],
      "inputs": [
        {
          "name": "evaluation_results",
          "type_hint": "List[EvaluationResult]",
          "default_value": None,
          "is_bound": False,
          "is_fulfilled_by_sibling": False
        }
      ]
    }
  ],
  "edges": [
    {
      "id": "e_4634062064__4633915232_4634062064__4633684608__4633688064_query",
      "source": "4634062064__4633915232",
      "target": "4634062064__4633684608__4633688064",
      "edge_type": "data_flow",
      "mapping_label": None,
      "source_level_id": "root__nested_4634062064",
      "target_level_id": "root__nested_4634062064__nested_4633684608"
    },
    {
      "id": "e_input_vector_store_4634062064__4633684608__4633688064",
      "source": "input_vector_store",
      "target": "4634062064__4633684608__4633688064",
      "edge_type": "parameter_flow",
      "mapping_label": None,
      "source_level_id": "root__nested_4634062064__nested_4633684608",
      "target_level_id": "root__nested_4634062064__nested_4633684608"
    },
    {
      "id": "e_input_top_k_4634062064__4633684608__4633688064",
      "source": "input_top_k",
      "target": "4634062064__4633684608__4633688064",
      "edge_type": "parameter_flow",
      "mapping_label": None,
      "source_level_id": "root__nested_4634062064__nested_4633684608",
      "target_level_id": "root__nested_4634062064__nested_4633684608"
    },
    {
      "id": "e_4634062064__4633684608__4633688064_4634062064__4633684608__4633688016",
      "source": "4634062064__4633684608__4633688064",
      "target": "4634062064__4633684608__4633688016",
      "edge_type": "data_flow",
      "mapping_label": None,
      "source_level_id": "root__nested_4634062064__nested_4633684608",
      "target_level_id": "root__nested_4634062064__nested_4633684608"
    },
    {
      "id": "e_4634062064__4633915232_4634062064__4633684608__4633688016_query",
      "source": "4634062064__4633915232",
      "target": "4634062064__4633684608__4633688016",
      "edge_type": "data_flow",
      "mapping_label": None,
      "source_level_id": "root__nested_4634062064",
      "target_level_id": "root__nested_4634062064__nested_4633684608"
    },
    {
      "id": "e_input_llm_4634062064__4633684608__4633688016",
      "source": "input_llm",
      "target": "4634062064__4633684608__4633688016",
      "edge_type": "parameter_flow",
      "mapping_label": None,
      "source_level_id": "root__nested_4634062064__nested_4633684608",
      "target_level_id": "root__nested_4634062064__nested_4633684608"
    },
    {
      "id": "e_input_eval_pairs_4634062064__4633915232",
      "source": "input_eval_pairs",
      "target": "4634062064__4633915232",
      "edge_type": "parameter_flow",
      "mapping_label": "eval_pairs \u2192 eval_pair",
      "source_level_id": "root__nested_4634062064",
      "target_level_id": "root__nested_4634062064"
    },
    {
      "id": "e_4634062064__4633684608__4633688016_4634062064__4633913024_answer",
      "source": "4634062064__4633684608__4633688016",
      "target": "4634062064__4633913024",
      "edge_type": "data_flow",
      "mapping_label": None,
      "source_level_id": "root__nested_4634062064__nested_4633684608",
      "target_level_id": "root__nested_4634062064"
    },
    {
      "id": "e_input_eval_pairs_4634062064__4633913024",
      "source": "input_eval_pairs",
      "target": "4634062064__4633913024",
      "edge_type": "parameter_flow",
      "mapping_label": "eval_pairs \u2192 eval_pair",
      "source_level_id": "root__nested_4634062064",
      "target_level_id": "root__nested_4634062064"
    },
    {
      "id": "e_input_judge_4634062064__4633913024",
      "source": "input_judge",
      "target": "4634062064__4633913024",
      "edge_type": "parameter_flow",
      "mapping_label": None,
      "source_level_id": "root__nested_4634062064",
      "target_level_id": "root__nested_4634062064"
    },
    {
      "id": "e_4634062064__4633913024_4634243584_evaluation_results",
      "source": "4634062064__4633913024",
      "target": "4634243584",
      "edge_type": "data_flow",
      "mapping_label": "evaluation_result \u2192 evaluation_results",
      "source_level_id": "root__nested_4634062064",
      "target_level_id": "root"
    }
  ],
  "input_levels": {
    "vector_store": "root__nested_4634062064__nested_4633684608",
    "top_k": "root__nested_4634062064__nested_4633684608",
    "llm": "root__nested_4634062064__nested_4633684608",
    "eval_pairs": "root",
    "judge": "root__nested_4634062064"
  }
}

# Transform and generate HTML for Depth 1
react_flow_data_d1 = transform_to_react_flow(graph_data, initial_depth=1)
html_content_d1 = generate_widget_html(react_flow_data_d1)

with open("complex_viz_depth1.html", "w") as f:
    f.write(html_content_d1)

print("Generated complex_viz_depth1.html")

# Transform and generate HTML for Depth 2
react_flow_data_d2 = transform_to_react_flow(graph_data, initial_depth=2)
html_content_d2 = generate_widget_html(react_flow_data_d2)

with open("complex_viz_depth2.html", "w") as f:
    f.write(html_content_d2)

print("Generated complex_viz_depth2.html")
