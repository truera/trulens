| Attribute/Namespace | <div style="width:500px">Meaning</div> | When Required? | Is Namespace? | Type |
|:--------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------|:------------------------------|:----------------------|
| `ai.observability.span_type` | Span type. This states what kind of span this is. E.g. "retrieval", "generation", "unknown", "record root". Given a span type, we can assume there might be relevant fields in `ai.observability.<span type>`. For example, for a span of type "record_root", there'll be more span attributes in the namespace `ai.observability.record_root` | Never | | str |
| `ai.observability.record_id` | Record ID. This ties all spans of a single invocation to the app together. We don't use the trace id for this purpose because a trace may have multiple records (i.e. app invocations). | Always | | str |
| `ai.observability.app_id` | App ID. | Always | | str |
| `ai.observability.app_name` | App name. | Always | | str |
| `ai.observability.app_version` | App version. | Always | | str |
| `ai.observability.run.name` | Run name. Runs represent a set of invocations to the app. | Always for Snowflake for non-evaluation spans | | str |
| `ai.observability.input_id` | ID of the input to the app for this record. | Always for Snowflake for non-evaluation spans | | str |
| `ai.observability.span_groups` | List of groups that the span belongs to. This is primarily used for metric computation. | Never | | str \| List[str] |
| `ai.observability.record_root` | Namespace for attributes specific to the record root. | | Y | |
| `ai.observability.record_root.input` | Main input to the app for this record. | Never | | Any (but usually str) |
| `ai.observability.record_root.output` | Main output to the app for this record. | Never | | Any (but usually str) |
| `ai.observability.record_root.error` | Error thrown by app for this record. Exclusive with main output. | Never | | Any (but usually str) |
| `ai.observability.record_root.ground_truth_output` | Ground truth of the record. | Never | | Any (but usually str) |
| `ai.observability.eval_root` | Namespace for attributes specific to the root span of a feedback evaluation. | Never | Y | |
| `ai.observability.eval_root.metric_name` | Name of the feedback definition being evaluated. | Always for eval_root spans | | str |
| `ai.observability.eval_root.span_group` | Span group of the inputs to this metric. | Never | | str |
| `ai.observability.eval_root.args_metadata.span_id` | Mapping of argument name of the feedback function to the ID of the span that provided it. E.g. if the feedback function has an input `x` that came from a span with id "123", then `ai.observability.eval_root.args_metadata.span_id.x` will have value "123". | Always for evaluation root spans | Y | str -> str |
| `ai.observability.eval_root.args_metadata.span_attribute` | Mapping of argument name of the feedback function to the attribute of the span that provided it. E.g. if the feedback function has an input `x` that came from a span attribute "abc", then `ai.observability.eval_root.args_metadata.span_attribute.x` will have value "abc". | Never | Y | str -> str |
| `ai.observability.eval_root.error` | Error raised during evaluation. | Never | | Any (but usually str) |
| `ai.observability.eval_root.score` | Score of the evaluation. | Always for evaluation root spans | | float |
| `ai.observability.eval_root.higher_is_better` | Whether higher is better for this feedback function. | Never | | bool |
| `ai.observability.eval_root.metadata` | Any other metadata of the evaluation. | Never | Y | str -> Any |
| `ai.observability.eval` | Namespace for attributes specific to feedback function evaluation spans. | | Y | |
| `ai.observability.eval.target_record_id` | Record id of the record being evaluated. | Never | | str |
| `ai.observability.eval.eval_root_id` | Span id for the "eval_root" span this span is under. | Always for eval or eval_root spans | | str |
| `ai.observability.eval.criteria` | Criteria for this sub-step. | Never | | str |
| `ai.observability.eval.explanation` | Explanation for the score for this sub-step. | Never | | str |
| `ai.observability.eval.score` | Score for this sub-step. | Never | | float |
| `ai.observability.cost` | Namespace for cost information. | Never | Y | |
| `ai.observability.cost.cost` | Cost. | Never | | float |
| `ai.observability.cost.cost_currency` | Currency of the cost. | Never | | str |
| `ai.observability.cost.model` | Model used that caused any costs. | Never | | str |
| `ai.observability.cost.num_tokens` | Total tokens processed. | Never | | int |
| `ai.observability.cost.num_prompt_tokens` | Number of prompt tokens supplied. | Never | | int |
| `ai.observability.cost.num_completion_tokens` | Number of completion tokens generated. | Never | | int |
| `ai.observability.call` | Namespace for instrumented method call attributes. | | Y | |
| `ai.observability.call.function` | Name of function being tracked. | Never | | str |
| `ai.observability.call.kwargs` | Namespace from function's argument name to value. E.g. if the function has a parameter `x` whose value was "y", then we'd have `ai.observability.call.kwargs.x` have value "y". | Never | Y | str -> Any |
| `ai.observability.call.return` | Return value of the function if it executed without error. | Never | | Any |
| `ai.observability.call.error` | Error raised by the function if it executed with an error. | Never | | Any (but usually str) |
| `ai.observability.retrieval` | Namespace for attributes specific to a retrieval span. | | Y | |
| `ai.observability.retrieval.query_text` | Input text whose related contexts are being retrieved. | Never | | str |
| `ai.observability.retrieval.num_contexts` | The number of contexts requested, not necessarily retrieved. | Never | | int |
| `ai.observability.retrieval.retrieved_contexts` | The retrieved contexts. | Never | | List[str] |
| `ai.observability.generation` | Namespace for attributes specific to a generation span. | | Y | |
| `ai.observability.graph_task` | Namespace for attributes specific to a graph task function execution span. | | Y | |
| `ai.observability.graph_task.task_name` | Name of the task function. | Never | | str |
| `ai.observability.graph_task.input_state` | Input state to the task. | Never | | Any |
| `ai.observability.graph_task.output_state` | Output state from the task. | Never | | Any |
| `ai.observability.graph_task.error` | Error raised during task execution. | Never | | Any (but usually str) |
| `ai.observability.graph_node` | Namespace for attributes specific to a graph node execution span. | | Y | |
| `ai.observability.graph_node.node_name` | Name of the node. | Never | | str |
| `ai.observability.graph_node.input_state` | Input state to the graph. | Never | | Any |
| `ai.observability.graph_node.output_state` | Output state from the graph. | Never | | Any |
| `ai.observability.graph_node.latest_message` | Latest message flowing between nodes. | Never | | Any |
| `ai.observability.graph_node.nodes_executed` | List of nodes executed in the graph. | Never | | List[str] |
| `ai.observability.graph_node.error` | Error raised during graph execution. | Never | | Any (but usually str) |
| `ai.observability.workflow` | Namespace for attributes specific to a workflow execution span. | | Y | |
| `ai.observability.workflow.input_event` | Input event to the workflow. | Never | | Any |
| `ai.observability.workflow.output_event` | Output event from the workflow. | Never | | Any |
| `ai.observability.workflow.error` | Error raised during workflow execution. | Never | | Any (but usually str) |
| `ai.observability.workflow.agent_name` | Name of the agent executing in the workflow. | Never | | str |
| `ai.observability.agent` | Namespace for attributes specific to an agent execution span. | | Y | |
| `ai.observability.tool` | Namespace for attributes specific to a tool/function call execution span. | | Y | |
| `ai.observability.reranking` | Namespace for attributes specific to a reranking operation span. | | Y | |
| `ai.observability.reranking.query_text` | Query text used for reranking. | Never | | str |
| `ai.observability.reranking.model_name` | Name of the reranking model. | Never | | str |
| `ai.observability.reranking.top_n` | Number of top results to return after reranking. | Never | | int |
| `ai.observability.reranking.input_context_texts` | Input contexts before reranking. | Never | | List[str] |
| `ai.observability.reranking.input_context_scores` | Input scores before reranking. | Never | | List[float] |
| `ai.observability.reranking.input_ranks` | Input ranking order before reranking. | Never | | List[int] |
| `ai.observability.reranking.output_ranks` | Output ranking order after reranking. | Never | | List[int] |
| `ai.observability.reranking.output_context_texts` | Output contexts after reranking. | Never | | List[str] |
| `ai.observability.reranking.output_context_scores` | Output scores after reranking. | Never | | List[float] |
