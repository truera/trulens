/**
 * Semantic conventions for TruLens spans.
 *
 * Direct TypeScript port of trulens/otel/semconv/trace.py.
 * No runtime dependencies — safe to import in any context.
 *
 * Relevant links:
 * - https://opentelemetry.io/docs/specs/semconv/
 * - https://opentelemetry.io/docs/specs/semconv/gen-ai/
 */

const BASE_SCOPE = "ai.observability";

export const ResourceAttributes = {
  /** ID of the app that the span belongs to. */
  APP_ID: `${BASE_SCOPE}.app_id`,
  /** Fully qualified name of the app that the span belongs to. */
  APP_NAME: `${BASE_SCOPE}.app_name`,
  /** Name of the version that the span belongs to. */
  APP_VERSION: `${BASE_SCOPE}.app_version`,
} as const;

/** Span type attribute values. */
export enum SpanType {
  UNKNOWN = "unknown",
  RECORD_ROOT = "record_root",
  EVAL_ROOT = "eval_root",
  EVAL = "eval",
  RETRIEVAL = "retrieval",
  GENERATION = "generation",
  GRAPH_TASK = "graph_task",
  GRAPH_NODE = "graph_node",
  WORKFLOW_STEP = "workflow_step",
  AGENT = "agent",
  TOOL = "tool",
  RERANKER = "reranking",
  MCP = "MCP",
}

export const SpanAttributes = {
  /** Span type attribute key. */
  SPAN_TYPE: `${BASE_SCOPE}.span_type`,
  /** ID of the record that the span belongs to. */
  RECORD_ID: `${BASE_SCOPE}.record_id`,
  /** Name of the run that the span belongs to. */
  RUN_NAME: `${BASE_SCOPE}.run.name`,
  /** ID of the input that the span belongs to. */
  INPUT_ID: `${BASE_SCOPE}.input_id`,
  /** Number of records processed in a run. */
  INPUT_RECORDS_COUNT: `${BASE_SCOPE}.input_records_count`,
  /** List of groups that the span belongs to. */
  SPAN_GROUPS: `${BASE_SCOPE}.span_groups`,

  RECORD_ROOT: {
    base: `${BASE_SCOPE}.record_root`,
    /** Main input to the app. */
    INPUT: `${BASE_SCOPE}.record_root.input`,
    /** Main output of the app. */
    OUTPUT: `${BASE_SCOPE}.record_root.output`,
    /** Main error of the app (exclusive with OUTPUT). */
    ERROR: `${BASE_SCOPE}.record_root.error`,
    /** Ground truth of the record. */
    GROUND_TRUTH_OUTPUT: `${BASE_SCOPE}.record_root.ground_truth_output`,
  },

  EVAL_ROOT: {
    base: `${BASE_SCOPE}.eval_root`,
    /** Name of the feedback definition being evaluated. */
    METRIC_NAME: `${BASE_SCOPE}.eval_root.metric_name`,
    /** The span group of the inputs to this metric. */
    SPAN_GROUP: `${BASE_SCOPE}.eval_root.span_group`,
    /** Scope: mapping of argument name to span ID. */
    ARGS_SPAN_ID: `${BASE_SCOPE}.eval_root.args_metadata.span_id`,
    /** Scope: mapping of argument name to span attribute name. */
    ARGS_SPAN_ATTRIBUTE: `${BASE_SCOPE}.eval_root.args_metadata.span_attribute`,
    /** Error raised during evaluation. */
    ERROR: `${BASE_SCOPE}.eval_root.error`,
    /** Score of the evaluation. */
    SCORE: `${BASE_SCOPE}.eval_root.score`,
    /** Whether higher is better for this feedback function. */
    HIGHER_IS_BETTER: `${BASE_SCOPE}.eval_root.higher_is_better`,
    /** Explanation for the score. */
    EXPLANATION: `${BASE_SCOPE}.eval_root.explanation`,
    /** Any metadata of the evaluation. */
    METADATA: `${BASE_SCOPE}.eval_root.metadata`,
  },

  EVAL: {
    base: `${BASE_SCOPE}.eval`,
    /** Record id of the record being evaluated. */
    TARGET_RECORD_ID: `${BASE_SCOPE}.eval.target_record_id`,
    /** Span id for the EVAL_ROOT span this span is under. */
    EVAL_ROOT_ID: `${BASE_SCOPE}.eval.eval_root_id`,
    /** Name of the feedback definition being evaluated. */
    METRIC_NAME: `${BASE_SCOPE}.eval.metric_name`,
    /** Criteria for this sub-step. */
    CRITERIA: `${BASE_SCOPE}.eval.criteria`,
    /** Explanation for the score for this sub-step. */
    EXPLANATION: `${BASE_SCOPE}.eval.explanation`,
    /** Any metadata for this sub-step. */
    METADATA: `${BASE_SCOPE}.eval.metadata`,
    /** Score for this sub-step. */
    SCORE: `${BASE_SCOPE}.eval.score`,
    /** Error raised during this sub-step. */
    ERROR: `${BASE_SCOPE}.eval.error`,
  },

  COST: {
    base: `${BASE_SCOPE}.cost`,
    /** Cost of the span. */
    COST: `${BASE_SCOPE}.cost.cost`,
    /** Currency of the cost. */
    CURRENCY: `${BASE_SCOPE}.cost.cost_currency`,
    /** Model used that caused any costs. */
    MODEL: `${BASE_SCOPE}.cost.model`,
    /** Total tokens processed. */
    NUM_TOKENS: `${BASE_SCOPE}.cost.num_tokens`,
    /** Number of prompt tokens supplied. */
    NUM_PROMPT_TOKENS: `${BASE_SCOPE}.cost.num_prompt_tokens`,
    /** Number of completion tokens generated. */
    NUM_COMPLETION_TOKENS: `${BASE_SCOPE}.cost.num_completion_tokens`,
    /** Number of reasoning tokens generated (for reasoning models). */
    NUM_REASONING_TOKENS: `${BASE_SCOPE}.cost.num_reasoning_tokens`,
  },

  CALL: {
    base: `${BASE_SCOPE}.call`,
    /** Name of function being tracked. */
    FUNCTION: `${BASE_SCOPE}.call.function`,
    /** Scope: keyword arguments of the function. */
    KWARGS: `${BASE_SCOPE}.call.kwargs`,
    /** Return value of the function. */
    RETURN: `${BASE_SCOPE}.call.return`,
    /** Error raised by the function. */
    ERROR: `${BASE_SCOPE}.call.error`,
  },

  RETRIEVAL: {
    base: `${BASE_SCOPE}.retrieval`,
    /** Input text whose related contexts are being retrieved. */
    QUERY_TEXT: `${BASE_SCOPE}.retrieval.query_text`,
    /** The number of contexts requested. */
    NUM_CONTEXTS: `${BASE_SCOPE}.retrieval.num_contexts`,
    /** The retrieved contexts. */
    RETRIEVED_CONTEXTS: `${BASE_SCOPE}.retrieval.retrieved_contexts`,
  },

  GENERATION: {
    base: `${BASE_SCOPE}.generation`,
  },

  GRAPH_TASK: {
    base: `${BASE_SCOPE}.graph_task`,
    /** Name of the task function. */
    TASK_NAME: `${BASE_SCOPE}.graph_task.task_name`,
    /** Input state to the task. */
    INPUT_STATE: `${BASE_SCOPE}.graph_task.input_state`,
    /** Output state from the task. */
    OUTPUT_STATE: `${BASE_SCOPE}.graph_task.output_state`,
    /** Error raised during task execution. */
    ERROR: `${BASE_SCOPE}.graph_task.error`,
  },

  GRAPH_NODE: {
    base: `${BASE_SCOPE}.graph_node`,
    /** Name of the node. */
    NODE_NAME: `${BASE_SCOPE}.graph_node.node_name`,
    /** Input state to the graph. */
    INPUT_STATE: `${BASE_SCOPE}.graph_node.input_state`,
    /** Output state from the graph. */
    OUTPUT_STATE: `${BASE_SCOPE}.graph_node.output_state`,
    /** Latest message flowing between nodes. */
    LATEST_MESSAGE: `${BASE_SCOPE}.graph_node.latest_message`,
    /** List of nodes executed in the graph. */
    NODES_EXECUTED: `${BASE_SCOPE}.graph_node.nodes_executed`,
    /** Error raised during graph execution. */
    ERROR: `${BASE_SCOPE}.graph_node.error`,
  },

  WORKFLOW: {
    base: `${BASE_SCOPE}.workflow`,
    /** Input event to the workflow. */
    INPUT_EVENT: `${BASE_SCOPE}.workflow.input_event`,
    /** Output event from the workflow. */
    OUTPUT_EVENT: `${BASE_SCOPE}.workflow.output_event`,
    /** Error raised during workflow execution. */
    ERROR: `${BASE_SCOPE}.workflow.error`,
    /** Name of the agent executing in the workflow. */
    AGENT_NAME: `${BASE_SCOPE}.workflow.agent_name`,
  },

  RERANKER: {
    base: `${BASE_SCOPE}.reranking`,
    /** Query text used for reranking. */
    QUERY_TEXT: `${BASE_SCOPE}.reranking.query_text`,
    /** Name of the reranking model. */
    MODEL_NAME: `${BASE_SCOPE}.reranking.model_name`,
    /** Number of top results to return after reranking. */
    TOP_N: `${BASE_SCOPE}.reranking.top_n`,
    /** Input contexts before reranking. */
    INPUT_CONTEXT_TEXTS: `${BASE_SCOPE}.reranking.input_context_texts`,
    /** Input scores before reranking. */
    INPUT_CONTEXT_SCORES: `${BASE_SCOPE}.reranking.input_context_scores`,
    /** Input ranking order before reranking. */
    INPUT_RANKS: `${BASE_SCOPE}.reranking.input_ranks`,
    /** Output ranking order after reranking. */
    OUTPUT_RANKS: `${BASE_SCOPE}.reranking.output_ranks`,
    /** Output contexts after reranking. */
    OUTPUT_CONTEXT_TEXTS: `${BASE_SCOPE}.reranking.output_context_texts`,
    /** Output scores after reranking. */
    OUTPUT_CONTEXT_SCORES: `${BASE_SCOPE}.reranking.output_context_scores`,
  },

  MCP: {
    base: `${BASE_SCOPE}.mcp`,
    /** Name of the MCP tool being called. */
    TOOL_NAME: `${BASE_SCOPE}.mcp.tool_name`,
    /** Description of the MCP tool. */
    TOOL_DESCRIPTION: `${BASE_SCOPE}.mcp.tool_description`,
    /** Name of the MCP server providing the tool. */
    SERVER_NAME: `${BASE_SCOPE}.mcp.server_name`,
    /** Schema of the input parameters for the MCP tool. */
    INPUT_SCHEMA: `${BASE_SCOPE}.mcp.input_schema`,
    /** Arguments passed to the MCP tool. */
    INPUT_ARGUMENTS: `${BASE_SCOPE}.mcp.input_arguments`,
    /** Content returned by the MCP tool. */
    OUTPUT_CONTENT: `${BASE_SCOPE}.mcp.output_content`,
    /** Whether the MCP tool call resulted in an error. */
    OUTPUT_IS_ERROR: `${BASE_SCOPE}.mcp.output_is_error`,
    /** Time taken to execute the MCP tool call in milliseconds. */
    EXECUTION_TIME_MS: `${BASE_SCOPE}.mcp.execution_time_ms`,
  },

  INLINE_EVAL: {
    base: `${BASE_SCOPE}.inline_eval`,
    /** Boolean flag indicating whether a span should be exported. */
    EMIT_SPAN: `${BASE_SCOPE}.inline_eval.emit_span`,
  },
} as const;
