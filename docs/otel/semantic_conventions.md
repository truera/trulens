# Semantic Conventions

TruLens traces combine standard OpenTelemetry conventions with TruLens-owned
extensions. Use an official OpenTelemetry convention whenever one exists. Use
the `ai.observability.*` namespace only for record, evaluation, agentic, and
framework concepts that OpenTelemetry does not define.

## Convention ownership

| Namespace or field | Owner | Standard | Usage |
|---|---|---|---|
| `gen_ai.*` | OpenTelemetry | [Generative AI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) | Model inference, structured messages, token usage, and tool execution. Do not add TruLens-specific fields to this namespace. |
| `error.type` | OpenTelemetry | [Recording errors](https://opentelemetry.io/docs/specs/semconv/general/recording-errors/) | Stable error class or category on failed spans, paired with OTEL error status. |
| `service.name`, `service.version` | OpenTelemetry | [Resource semantic conventions](https://opentelemetry.io/docs/specs/semconv/resource/) | Application identity on the OTEL resource. |
| `ai.observability.*` | TruLens | TruLens semantic conventions | Record/evaluation identity, selectors, costs, framework spans, and concepts without an OTEL equivalent. |
| `ai.observability.coding_agent.*` | TruLens extension | No OTEL coding-agent convention currently exists | Coding client, native lifecycle event, editor version, workspace, and privacy-controlled diffs. |
| `ai.observability.mcp.*` | TruLens extension | No OTEL MCP convention currently exists | MCP server identity, schemas, arguments, outputs, errors, and duration. MCP tool execution also emits applicable official `gen_ai.tool.*` fields. |

`ai.observability.app_name` and `ai.observability.app_version` are TruLens
application-routing attributes despite the historical `ResourceAttributes`
Python class name. Instrumentation also places `service.name` and
`service.version` on the actual OpenTelemetry resource.

## Semantic convention reference

All conventions use the same reference columns:

- **Convention**: Attribute, event, or namespace.
- **Meaning**: Semantic meaning of the value.
- **Emitted when**: Exact condition under which TruLens emits it.
- **Scope**: Span, event, resource, or namespace carrying the value.
- **Type**: Emitted value type.

### OpenTelemetry conventions

TruLens emits these official OpenTelemetry fields alongside its own record and
evaluation attributes. Allowed values are governed by the linked OpenTelemetry
specification.

| Convention | <div style="width:500px">Meaning</div> | Emitted when | Scope | Type |
|:---|:---|:---|:---|:---|
| `gen_ai.operation.name` | GenAI operation represented by the span. | On each recognized model inference or tool execution. Current values are `chat` and `execute_tool`. | Generation, tool, and MCP spans | str |
| `gen_ai.request.model` | Model requested by the client. | The instrumentation observes a request-side model identifier. | Generation span | str |
| `gen_ai.response.model` | Model that produced the response. | A response-bearing event reports the actual model. It is not copied from request data when response-side evidence is absent. | Generation span | str |
| `gen_ai.system` | Provider or inference system. | Provider identity is known from native metadata or can be inferred unambiguously from the model namespace. Coding clients such as Cursor are not emitted as providers. | Generation span | str |
| `gen_ai.usage.input_tokens` | Input tokens consumed by inference. | At least one event reports input usage; TruLens emits the latest reported value rather than summing cumulative hook reports. | Generation span | int |
| `gen_ai.usage.output_tokens` | Output tokens produced by inference. | At least one event reports output usage; TruLens emits the latest reported value. | Generation span | int |
| `gen_ai.tool.name` | Name of the executed tool. | A normalized hook event is categorized as a tool or MCP execution. | Tool and MCP spans | str |
| `gen_ai.tool.call.id` | Native identifier for one tool call. | The client supplies a tool, operation, or call ID. | Tool and MCP spans | str |
| `gen_ai.tool.call.arguments` | JSON-serialized tool arguments. | Tool-payload capture is enabled and arguments are available after privacy filtering. | Tool and MCP spans | str |
| `gen_ai.tool.call.result` | JSON-serialized tool result. | Tool-payload capture is enabled and a result is available after privacy filtering. | Tool and MCP spans | str |
| `gen_ai.client.inference.operation.details` | Event containing structured inference messages. | Content capture is enabled and a generation response is available. | Generation span event | event |
| `gen_ai.input.messages` | Structured input messages using `role` and `parts`. | The inference-details event is emitted and captured prompt content is available. Text parts use `{ "type": "text", "content": "..." }`. | Inference-details event | JSON str |
| `gen_ai.output.messages` | Structured output messages using `role` and `parts`. | The inference-details event is emitted and captured response content is available. | Inference-details event | JSON str |
| `error.type` | Stable error class or category. | A span has failed, including incomplete turns and client-hook/tool errors. | Failed span | str |
| OTEL span status `ERROR` | Indicates failed execution. | A normalized operation, terminal turn, or stale turn fails. The captured error is used as status description when available. | Failed span status | status |
| `service.name` | Application/service identity. | On every assembled hook span resource; defaults to the native client app name. | Resource | str |
| `service.version` | Native application version. | On every assembled hook span resource; uses the detected/declared client version or `unknown`. | Resource | str |

TruLens currently follows the `gen_ai.system` convention selected by its OTEL
compatibility baseline. `gen_ai.provider.name` exists in newer OTEL revisions,
but instrumentation must not emit both revisions or switch defaults without an
explicit compatibility change.

### TruLens conventions

Every attribute in the following table uses the TruLens-owned
`ai.observability.*` namespace. These are extensions to OpenTelemetry, not
official OpenTelemetry GenAI fields. Rows that say `Never` are legacy shorthand
for optional emission: the attribute is emitted when its source value is
available and allowed by capture policy. `Y` in **Scope** is legacy shorthand
for a namespace rather than a leaf attribute.

| Convention | <div style="width:500px">Meaning</div> | Emitted when | Scope | Type |
|:--------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------|:------------------------------|:----------------------|
| `ai.observability.span_type` | Span type. This states what kind of span this is. E.g. "retrieval", "generation", "unknown", "record root". Given a span type, we can assume there might be relevant fields in `ai.observability.<span type>`. For example, for a span of type "record_root", there'll be more span attributes in the namespace `ai.observability.record_root` | An instrumented span is assigned a TruLens semantic type. | Span | str |
| `ai.observability.record_id` | Record ID. This ties all spans of a single invocation to the app together. We don't use the trace id for this purpose because a trace may have multiple records (i.e. app invocations). | Every span belongs to a TruLens app invocation. | Span | str |
| `ai.observability.app_id` | App ID. | The registered application has an app ID. | Span | str |
| `ai.observability.app_name` | App name. | Every application span; Snowflake routing requires it. | Span | str |
| `ai.observability.app_version` | App version. | Every application span; Snowflake routing requires it. | Span | str |
| `ai.observability.run.name` | Run name. Runs represent a set of invocations to the app. | A run is active; required for Snowflake non-evaluation spans. | Span | str |
| `ai.observability.input_id` | ID of the input to the app for this record. | An input can be identified; required for Snowflake non-evaluation spans. | Span | str |
| `ai.observability.span_groups` | List of groups that the span belongs to. This is primarily used for metric computation. | The span is assigned to one or more metric-computation groups. | Span | str \| List[str] |
| `ai.observability.record_root` | Namespace for attributes specific to the record root. | Namespace only; never emitted as a leaf value. | Namespace | |
| `ai.observability.record_root.input` | Main input to the app for this record. | A record-root span is created; content is captured or replaced by the configured privacy placeholder. | Record-root span | Any (but usually str) |
| `ai.observability.record_root.output` | Main output to the app for this record. | The invocation returns or an incomplete/private-output placeholder is required. | Record-root span | Any (but usually str) |
| `ai.observability.record_root.error` | Error thrown by app for this record. Exclusive with main output. | The app invocation or incomplete turn fails. | Failed record-root span | Any (but usually str) |
| `ai.observability.record_root.ground_truth_output` | Ground truth of the record. | Ground-truth output is supplied for the record. | Record-root span | Any (but usually str) |
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
| `ai.observability.mcp` | Namespace for attributes specific to Model Context Protocol (MCP) tool calls. | Namespace only; never emitted as a leaf value. | Namespace | |
| `ai.observability.mcp.tool_name` | Name of the MCP tool being called. | An MCP span has a tool name. | MCP span | str |
| `ai.observability.mcp.tool_description` | Description of the MCP tool. | The MCP tool definition supplies a description. | MCP span | str |
| `ai.observability.mcp.server_name` | Name of the MCP server providing the tool. | The native event identifies the MCP server. | MCP span | str |
| `ai.observability.mcp.input_schema` | Schema of the input parameters for the MCP tool. | The MCP tool definition supplies an input schema. | MCP span | str |
| `ai.observability.mcp.input_arguments` | Arguments passed to the MCP tool. | Tool-payload capture is enabled and MCP arguments are available after privacy filtering. | MCP span | str |
| `ai.observability.mcp.output_content` | Content returned by the MCP tool. | Tool-payload capture is enabled and MCP output is available after privacy filtering. | MCP span | str |
| `ai.observability.mcp.output_is_error` | Whether the MCP tool call resulted in an error. | Every normalized MCP completion. | MCP span | bool |
| `ai.observability.mcp.execution_time_ms` | Time taken to execute the MCP tool call in milliseconds. | The client reports MCP execution duration. | MCP span | float |
| `ai.observability.coding_agent` | Namespace for coding-agent client extensions without an OTEL equivalent. | Namespace only; never emitted as a leaf value. | Namespace | |
| `ai.observability.coding_agent.client` | Native coding client, such as Cursor, Claude Code, or OpenCode. | Every normalized coding-agent operation span. | Tool, MCP, agent, or workflow span | str |
| `ai.observability.coding_agent.native_event` | Native lifecycle event reported by the coding client. | Every normalized coding-agent operation span. | Tool, MCP, agent, or workflow span | str |
| `ai.observability.coding_agent.diff` | Privacy-controlled source diff or old/new edit pairs. | Diff capture is enabled and a native edit or patch contains a diff. | Edit/tool span | str |
| `ai.observability.coding_agent.editor_version` | Native editor or CLI version. | The native event or launcher reports a client version. | Coding-agent operation span | str |
| `ai.observability.coding_agent.workspace` | Privacy-controlled workspace path or identifier. | Path capture is enabled and the event reports a workspace or working directory. | Coding-agent operation span | str |
| `ai.observability.generation` | Namespace for attributes specific to a generation span. | | Y | |
| `ai.observability.generation.is_streaming` | Whether the generation was streamed back incrementally. | Never | | bool |
| `ai.observability.generation.time_to_first_token_ms` | Milliseconds spent waiting for the first chunk of a streamed generation. | Never | | float |
| `ai.observability.generation.tokens_per_second` | Completion tokens generated per second, measured from the first chunk to the last so that it reflects generation throughput rather than including time to first token. Only set when the completion token count is known, which for a stream means the request asked for usage to be included. | Never | | float |
| `ai.observability.generation.chunks_received` | Number of chunks received over a streamed generation. | Never | | int |
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
