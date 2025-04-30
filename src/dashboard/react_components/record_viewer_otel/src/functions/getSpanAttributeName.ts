import { BASE_SCOPE, SpanAttributes } from '../constants/span';
import { splitStringToTwoPartsByDelimiter } from './splitStringToTwoPartsByDelimiter';

export const getSpanAttributeName = (key: string) => {
  const mapping: Record<string, string> = {
    [SpanAttributes.RUN_NAME]: 'Run Name',
    [SpanAttributes.SPAN_TYPE]: 'Span Type',
    [SpanAttributes.SELECTOR_NAME]: 'Selector Name',
    [SpanAttributes.RECORD_ID]: 'Record ID',
    [SpanAttributes.APP_NAME]: 'App Name',
    [SpanAttributes.APP_VERSION]: 'App Version',
    [SpanAttributes.ROOT_SPAN_ID]: 'Root Span ID',
    [SpanAttributes.INPUT_ID]: 'Input ID',
    [SpanAttributes.DOMAIN]: 'Domain',
    [SpanAttributes.COST_COST_CURRENCY]: 'Cost Currency',
    [SpanAttributes.COST_MODEL]: 'Model',
    [SpanAttributes.COST_NUM_TOKENS]: 'Number of Tokens',
    [SpanAttributes.COST_NUM_PROMPT_TOKENS]: 'Number of Prompt Tokens',
    [SpanAttributes.COST_NUM_COMPLETION_TOKENS]: 'Number of Completion Tokens',
    [SpanAttributes.COST_NUM_GUARDRAIL_TOKENS]: 'Number of Guardrail Tokens',
    [SpanAttributes.COST_NUM_CORTEX_GUARDRAIL_TOKENS]: 'Number of Cortex Guardrail Tokens',
    [SpanAttributes.RECORDING_APP_ID]: 'App ID',
    [SpanAttributes.RECORD_ROOT_INPUT]: 'Input',
    [SpanAttributes.RECORD_ROOT_OUTPUT]: 'Output',
    [SpanAttributes.RECORD_ROOT_ERROR]: 'Error',
    [SpanAttributes.RECORD_ROOT_GROUND_TRUTH_OUTPUT]: 'Ground Truth',
    [SpanAttributes.EVAL_ROOT_TARGET_TRACE_ID]: 'Target Trace ID',
    [SpanAttributes.EVAL_ROOT_TARGET_SPAN_ID]: 'Target Span ID',
    [SpanAttributes.EVAL_ROOT_FEEDBACK_DEFINITION_ID]: 'Feedback Definition ID',
    [SpanAttributes.EVAL_ROOT_STATUS]: 'Status',
    [SpanAttributes.EVAL_ROOT_TOTAL_COST]: 'Total Cost',
    [SpanAttributes.EVAL_ROOT_ERROR]: 'Error',
    [SpanAttributes.EVAL_ROOT_SCORE]: 'Score',
    [SpanAttributes.EVAL_EVAL_ROOT_ID]: 'Evaluation Root ID',
    [SpanAttributes.EVAL_FEEDBACK_NAME]: 'Metric Name',
    [SpanAttributes.EVAL_ARGS]: 'Evaluation Arguments',
    [SpanAttributes.EVAL_CRITERIA]: 'Evaluation Criteria',
    [SpanAttributes.EVAL_TARGET_RECORD_ID]: 'Target Record ID',
    [SpanAttributes.EVAL_EXPLANATION]: 'Explanation',
    [SpanAttributes.EVAL_SCORE]: 'Evaluation Score',
    [SpanAttributes.EVAL_ROOT_METADATA]: 'Metadata',
    [SpanAttributes.COST_COST]: 'Cost',
    [SpanAttributes.CALL_CALL_ID]: 'Call ID',
    [SpanAttributes.CALL_STACK]: 'Call Stack',
    [SpanAttributes.CALL_SIGNATURE]: 'Call Signature',
    [SpanAttributes.CALL_FUNCTION]: 'Function',
    [SpanAttributes.CALL_CLASS]: 'Call Class',
    [SpanAttributes.CALL_ARGS]: 'Input',
    [SpanAttributes.CALL_KWARGS]: 'Input',
    [SpanAttributes.CALL_BOUND_ARGUMENTS]: 'Input',
    [SpanAttributes.CALL_RETURN]: 'Output',
    [SpanAttributes.CALL_ERROR]: 'Error',
    [SpanAttributes.RETRIEVAL_QUERY_TEXT]: 'Query Text',
    [SpanAttributes.RETRIEVAL_QUERY_EMBEDDING]: 'Query Embedding',
    [SpanAttributes.RETRIEVAL_DISTANCE_TYPE]: 'Distance Type',
    [SpanAttributes.RETRIEVAL_NUM_CONTEXTS]: 'Number of Contexts',
    [SpanAttributes.RETRIEVAL_RETRIEVED_CONTEXTS]: 'Retrieved Contexts',
    [SpanAttributes.RETRIEVAL_RETRIEVED_CONTEXT]: 'Retrieved Context',
    [SpanAttributes.RETRIEVAL_RETRIEVED_SCORES]: 'Retrieved Scores',
    [SpanAttributes.RETRIEVAL_RETRIEVED_EMBEDDINGS]: 'Retrieved Embeddings',
    [SpanAttributes.RERANKING_QUERY_TEXT]: 'Reranking Query Text',
    [SpanAttributes.RERANKING_MODEL_NAME]: 'Reranking Model Name',
    [SpanAttributes.RERANKING_TOP_N]: 'Top N',
    [SpanAttributes.RERANKING_INPUT_CONTEXT_TEXTS]: 'Reranking Input Context Texts',
    [SpanAttributes.RERANKING_INPUT_CONTEXT_SCORES]: 'Reranking Input Context Scores',
    [SpanAttributes.RERANKING_OUTPUT_RANKS]: 'Reranking Output Ranks',
    [SpanAttributes.GENERATION_MODEL_NAME]: 'Generation Model Name',
    [SpanAttributes.GENERATION_MODEL_TYPE]: 'Generation Model Type',
    [SpanAttributes.GENERATION_INPUT_TOKEN_COUNT]: 'Generation Input Token Count',
    [SpanAttributes.GENERATION_INPUT_MESSAGES]: 'Generation Input Messages',
    [SpanAttributes.GENERATION_OUTPUT_TOKEN_COUNT]: 'Generation Output Token Count',
    [SpanAttributes.GENERATION_OUTPUT_MESSAGES]: 'Generation Output Messages',
    [SpanAttributes.GENERATION_TEMPERATURE]: 'Generation Temperature',
    [SpanAttributes.GENERATION_COST]: 'Generation Cost',
    [SpanAttributes.MEMORIZATION_MEMORY_TYPE]: 'Memory Type',
    [SpanAttributes.MEMORIZATION_REMEMBERED]: 'Remembered',
    [SpanAttributes.EMBEDDING_INPUT_TEXT]: 'Embedding Input Text',
    [SpanAttributes.EMBEDDING_MODEL_NAME]: 'Embedding Model Name',
    [SpanAttributes.EMBEDDING_EMBEDDING]: 'Embedding',
    [SpanAttributes.TOOL_INVOCATION_DESCRIPTION]: 'Tool Invocation Description',
    [SpanAttributes.AGENT_INVOCATION_DESCRIPTION]: 'Agent Invocation Description',
  } as const;

  if (mapping[key]) return mapping[key];

  const keyOptions = splitStringToTwoPartsByDelimiter(key, '.');

  for (const [prefix, suffix] of keyOptions) {
    if (!mapping[prefix]) continue;

    return `${mapping[prefix]} (${suffix})`;
  }

  return key.replace(`${BASE_SCOPE}.`, '');
};
