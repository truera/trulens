/**
 * This file contains constants for the span attributes that are used for
 * AI Observability. Generally, these attributes should be kept in sync with
 * https://github.com/truera/trulens/blob/main/src/otel/semconv/trulens/otel/semconv/trace.py,
 * though code using these attributes should handle unknown attributes gracefully, to ensure
 * backwards and forwards compatibility.
 */

export const BASE_SCOPE = 'ai.observability';

/**
 * Span type, as recorded in "ai.observability.span_type" attribute within
 * the OTEL event table.
 */
export const SpanType = {
  UNKNOWN: 'unknown',
  CUSTOM: 'custom',
  RECORDING: 'recording',
  SEMANTIC: 'semantic',
  RECORD_ROOT: 'record_root',
  EVAL_ROOT: 'eval_root',
  EVAL: 'eval',
  RECORD: 'record',
  RETRIEVAL: 'retrieval',
  RERANKING: 'reranking',
  GENERATION: 'generation',
  MEMORIZATION: 'memorization',
  EMBEDDING: 'embedding',
  TOOL_INVOCATION: 'tool_invocation',
  AGENT_INVOCATION: 'agent_invocation',
} as const;

export type AIObservabilitySpanType = (typeof SpanType)[keyof typeof SpanType];

export const SpanAttributes = {
  RUN_NAME: `${BASE_SCOPE}.run.name`,

  // Attributes that are common to spans within a record
  SPAN_TYPE: `${BASE_SCOPE}.span_type`,
  SELECTOR_NAME: `${BASE_SCOPE}.selector_name`,
  RECORD_ID: `${BASE_SCOPE}.record_id`,
  APP_NAME: `${BASE_SCOPE}.app_name`,
  APP_VERSION: `${BASE_SCOPE}.app_version`,
  ROOT_SPAN_ID: `${BASE_SCOPE}.root_span_id`,
  INPUT_ID: `${BASE_SCOPE}.input_id`,
  DOMAIN: `${BASE_SCOPE}.domain`,

  // Attribute for cost-related information. These can apply regardless of the span type.
  COST_COST: `${BASE_SCOPE}.cost.cost`,
  COST_COST_CURRENCY: `${BASE_SCOPE}.cost.cost_currency`,
  COST_MODEL: `${BASE_SCOPE}.cost.model`,
  COST_NUM_TOKENS: `${BASE_SCOPE}.cost.num_tokens`,
  COST_NUM_PROMPT_TOKENS: `${BASE_SCOPE}.cost.num_prompt_tokens`,
  COST_NUM_COMPLETION_TOKENS: `${BASE_SCOPE}.cost.num_completion_tokens`,
  COST_NUM_GUARDRAIL_TOKENS: `${BASE_SCOPE}.cost.num_guardrail_tokens`,
  COST_NUM_CORTEX_GUARDRAIL_TOKENS: `${BASE_SCOPE}.cost.num_cortex_guardrails_tokens`,

  // Attributes for function calls. These can apply regardless of the span type.
  CALL_CALL_ID: `${BASE_SCOPE}.call.call_id`,
  CALL_STACK: `${BASE_SCOPE}.call.stack`,
  CALL_SIGNATURE: `${BASE_SCOPE}.call.signature`,
  CALL_FUNCTION: `${BASE_SCOPE}.call.function`,
  CALL_CLASS: `${BASE_SCOPE}.call.class`,
  CALL_ARGS: `${BASE_SCOPE}.call.args`,
  CALL_KWARGS: `${BASE_SCOPE}.call.kwargs`,
  CALL_BOUND_ARGUMENTS: `${BASE_SCOPE}.call.bound_arguments`,
  CALL_RETURN: `${BASE_SCOPE}.call.return`,
  CALL_ERROR: `${BASE_SCOPE}.call.error`,

  // Attributes for the RECORDING span type
  RECORDING_APP_ID: `${BASE_SCOPE}.${SpanType.RECORDING}.app_id`,

  // Attributes for the RECORD_ROOT span type
  RECORD_ROOT_GROUND_TRUTH_OUTPUT: `${BASE_SCOPE}.${SpanType.RECORD_ROOT}.ground_truth_output`,
  RECORD_ROOT_INPUT: `${BASE_SCOPE}.${SpanType.RECORD_ROOT}.input`,
  RECORD_ROOT_OUTPUT: `${BASE_SCOPE}.${SpanType.RECORD_ROOT}.output`,
  RECORD_ROOT_ERROR: `${BASE_SCOPE}.${SpanType.RECORD_ROOT}.error`,

  // Attributes for the EVAL_ROOT span type
  EVAL_ROOT_TARGET_TRACE_ID: `${BASE_SCOPE}.${SpanType.EVAL_ROOT}.target_trace_id`,
  EVAL_ROOT_TARGET_SPAN_ID: `${BASE_SCOPE}.${SpanType.EVAL_ROOT}.target_span_id`,
  EVAL_ROOT_FEEDBACK_DEFINITION_ID: `${BASE_SCOPE}.${SpanType.EVAL_ROOT}.feedback_definition_id`,
  EVAL_ROOT_STATUS: `${BASE_SCOPE}.${SpanType.EVAL_ROOT}.status`,
  EVAL_ROOT_TOTAL_COST: `${BASE_SCOPE}.${SpanType.EVAL_ROOT}.total_cost`,
  EVAL_ROOT_ERROR: `${BASE_SCOPE}.${SpanType.EVAL_ROOT}.error`,
  EVAL_ROOT_SCORE: `${BASE_SCOPE}.${SpanType.EVAL_ROOT}.score`,
  EVAL_ROOT_METADATA: `${BASE_SCOPE}.${SpanType.EVAL_ROOT}.metadata`,

  // Attributes for the EVAL span type
  EVAL_TARGET_RECORD_ID: `${BASE_SCOPE}.${SpanType.EVAL}.target_record_id`,
  EVAL_EVAL_ROOT_ID: `${BASE_SCOPE}.${SpanType.EVAL}.eval_root_id`,
  EVAL_FEEDBACK_NAME: `${BASE_SCOPE}.${SpanType.EVAL}.metric_name`,
  EVAL_ARGS: `${BASE_SCOPE}.${SpanType.EVAL}.args`,
  EVAL_CRITERIA: `${BASE_SCOPE}.${SpanType.EVAL}.criteria`,
  EVAL_EXPLANATION: `${BASE_SCOPE}.${SpanType.EVAL}.explanation`,
  EVAL_SCORE: `${BASE_SCOPE}.${SpanType.EVAL}.score`,

  // Attributes for RETRIEVAL span type
  RETRIEVAL_QUERY_TEXT: `${BASE_SCOPE}.${SpanType.RETRIEVAL}.query_text`,
  RETRIEVAL_QUERY_EMBEDDING: `${BASE_SCOPE}.${SpanType.RETRIEVAL}.query_embedding`,
  RETRIEVAL_DISTANCE_TYPE: `${BASE_SCOPE}.${SpanType.RETRIEVAL}.distance_type`,
  RETRIEVAL_NUM_CONTEXTS: `${BASE_SCOPE}.${SpanType.RETRIEVAL}.num_contexts`,
  RETRIEVAL_RETRIEVED_CONTEXTS: `${BASE_SCOPE}.${SpanType.RETRIEVAL}.retrieved_contexts`,
  RETRIEVAL_RETRIEVED_CONTEXT: `${BASE_SCOPE}.${SpanType.RETRIEVAL}.retrieved_context`,
  RETRIEVAL_RETRIEVED_SCORES: `${BASE_SCOPE}.${SpanType.RETRIEVAL}.retrieved_scores`,
  RETRIEVAL_RETRIEVED_EMBEDDINGS: `${BASE_SCOPE}.${SpanType.RETRIEVAL}.retrieved_embeddings`,

  // Attributes for RERANKING span type
  RERANKING_QUERY_TEXT: `${BASE_SCOPE}.${SpanType.RERANKING}.query_text`,
  RERANKING_MODEL_NAME: `${BASE_SCOPE}.${SpanType.RERANKING}.model_name`,
  RERANKING_TOP_N: `${BASE_SCOPE}.${SpanType.RERANKING}.top_n`,
  RERANKING_INPUT_CONTEXT_TEXTS: `${BASE_SCOPE}.${SpanType.RERANKING}.input_context_texts`,
  RERANKING_INPUT_CONTEXT_SCORES: `${BASE_SCOPE}.${SpanType.RERANKING}.input_context_scores`,
  RERANKING_OUTPUT_RANKS: `${BASE_SCOPE}.${SpanType.RERANKING}.output_ranks`,

  // Attributes for GENERATION span type
  GENERATION_MODEL_NAME: `${BASE_SCOPE}.${SpanType.GENERATION}.model_name`,
  GENERATION_MODEL_TYPE: `${BASE_SCOPE}.${SpanType.GENERATION}.model_type`,
  GENERATION_INPUT_TOKEN_COUNT: `${BASE_SCOPE}.${SpanType.GENERATION}.input_token_count`,
  GENERATION_INPUT_MESSAGES: `${BASE_SCOPE}.${SpanType.GENERATION}.input_messages`,
  GENERATION_OUTPUT_TOKEN_COUNT: `${BASE_SCOPE}.${SpanType.GENERATION}.output_token_count`,
  GENERATION_OUTPUT_MESSAGES: `${BASE_SCOPE}.${SpanType.GENERATION}.output_messages`,
  GENERATION_TEMPERATURE: `${BASE_SCOPE}.${SpanType.GENERATION}.temperature`,
  GENERATION_COST: `${BASE_SCOPE}.${SpanType.GENERATION}.cost`,

  // Attributes for MEMORIZATION span type
  MEMORIZATION_MEMORY_TYPE: `${BASE_SCOPE}.${SpanType.MEMORIZATION}.memory_type`,
  MEMORIZATION_REMEMBERED: `${BASE_SCOPE}.${SpanType.MEMORIZATION}.remembered`,

  // Attributes for EMBEDDING span type
  EMBEDDING_INPUT_TEXT: `${BASE_SCOPE}.${SpanType.EMBEDDING}.input_text`,
  EMBEDDING_MODEL_NAME: `${BASE_SCOPE}.${SpanType.EMBEDDING}.model_name`,
  EMBEDDING_EMBEDDING: `${BASE_SCOPE}.${SpanType.EMBEDDING}.embedding`,

  // Attributes for TOOL_INVOCATION span type
  TOOL_INVOCATION_DESCRIPTION: `${BASE_SCOPE}.${SpanType.TOOL_INVOCATION}.description`,

  // Attributes for AGENT_INVOCATION span type
  AGENT_INVOCATION_DESCRIPTION: `${BASE_SCOPE}.${SpanType.AGENT_INVOCATION}.description`,
} as const;
