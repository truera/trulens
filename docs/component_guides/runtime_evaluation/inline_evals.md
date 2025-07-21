# In-line Evaluations

In-line evaluations allow you to assess and score agent behavior as it happens—directly within the execution flow of your agent. Unlike post-hoc evaluations, which run after an agent completes its task, in-line evaluations provide real-time feedback by observing inputs, intermediate steps, or outputs during execution.

These evaluations can:

* Score individual steps such as retrieval or generation
* Detect recall issues, hallucinations or safety issues
* Affect agent orchestration by modifying the agent's state

By integrating evaluations into the runtime loop, agents can become more self-aware, adaptive, and robust, especially in complex or dynamic tasks.

`TruLens` inline evaluations perform two critical steps:

1. Execute an evaluation
2. Add the evaluation results to the agent's state

Consider a `LangGraph` agent with the following instrumented research node.

!!! example

    ```python
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes=lambda ret, exception, *args, **kwargs: {
            SpanAttributes.RETRIEVAL.QUERY_TEXT: args[0]["messages"][-1].content,
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [
                json.loads(dumps(message)).get("kwargs", {}).get("content", "")
                for message in ret.update["messages"]
                if isinstance(message, ToolMessage)
            ]
            if hasattr(ret, "update")
            else "No tool call",
        },
    )
    def research_node(
        state: MessagesState,
    ) -> Command[Literal["chart_generator", END]]:
        result = research_agent.invoke(state)
        goto = get_next_node(result["messages"][-1], "chart_generator")
        # wrap in a human message, as not all providers allow
        # AI message at the last position of the input messages list
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name="researcher"
        )
        return Command(
            update={
                # share internal message history of research agent with other agents
                "messages": result["messages"],
            },
            goto=goto,
        )
    ```

In this example, we can define a feedback function that accepts the `research_node`'s instrumented span attributes: `QUERY_TEXT` and `RETRIEVED_CONTEXTS`.

!!! example

    ```python
    f_context_relevance = (
        Feedback(
            provider.context_relevance_with_cot_reasons, name="Inline Context Relevance"
        )
        .on({
                "question": Selector(
                    span_type=SpanAttributes.SpanType.RETRIEVAL,
                    span_attribute=SpanAttributes.RETRIEVAL.QUERY_TEXT,
                )
            }
        )
        .on({
                "context": Selector(
                    span_type=SpanAttributes.SpanType.RETRIEVAL,
                    span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                    collect_list=False
                )
            }
        )
        .aggregate(np.mean)
    )
    ```

Then, once we have created a feedback function that operates on the instrumented span attributes for the method we want to evaluate, we can simply add the `@inline_evaluation` decorator with the feedback function we just created.

!!! example

    ```python
    @inline_evaluation(f_context_relevance)
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes=lambda ret, exception, *args, **kwargs: {
            SpanAttributes.RETRIEVAL.QUERY_TEXT: args[0]["messages"][-1].content,
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [
                json.loads(dumps(message)).get("kwargs", {}).get("content", "")
                for message in ret.update["messages"]
                if isinstance(message, ToolMessage)
            ]
            if hasattr(ret, "update")
            else "No tool call",
        },
    )
    def research_node(
        state: MessagesState,
    ) -> Command[Literal["chart_generator", END]]:
        result = research_agent.invoke(state)
        goto = get_next_node(result["messages"][-1], "chart_generator")
        # wrap in a human message, as not all providers allow
        # AI message at the last position of the input messages list
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name="researcher"
        )
        return Command(
            update={
                # share internal message history of research agent with other agents
                "messages": result["messages"],
            },
            goto=goto,
        )
    ```

!!! note

    Feedback functions used for inline evaluation must operate on available instrumented spans of the method that is being evaluated.

After the feedback function is executed, evaluation results will be added to the state. Inline evaluations are currently only implemented for `LangGraph`, additional framework support will follow.

!!! note "LangGraph-specific Implementation Details"

    In `LangGraph`, the evaluations are formatted as `AnyMessage` objects and appended to the `messages` key in `MessageState`.

By adding the evaluation results to the agent's state, the agent can then use evaluation results to guide execution steps. For example, by informing the agent that an initial retrieval step lacks context relevance, the agent may choose to perform additional research before moving on to generate a final answer.
