# Evals at Execution Time

In Evaluation a crucial role in improving output is produced by LLM apps by acting to change the application's execution flow at runtime.

TruLens supports inline evaluation via two different mechanisms:

1. In-line Evaluations - evaluations that are executed during an agent's execution flow and passed back to agent to assist in orchestration.
2. Guardrails - evaluations that can be used to block input, output and intermediate results produced by an application such as a RAG or agent.

## In-line Evaluations

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

## Guardrails

The second avenue for using evaluations to improve application output at execution time is via *guardrails*.

By setting guardrail thresholds based on feedback functions, we can directly leverage the same trusted evaluation metrics used for off-line observability, *at inference time*.

`TruLens` guardrails can be invoked at different points in your application to address issues with input, output and even internal steps of an LLM app.

### Output blocking guardrails

Typical guardrails *only* allow decisions based on the output, and have no impact on the intermediate steps of an LLM application.

![Output Blocking Guardrails Flow](simple_guardrail_flow.png)

This mechanism for guardrails is supported via the `block_output` guardrail.

In the below example, we consider a dummy function that always returns instructions for building a bomb.

Simply adding the `block_output` decorator with a feedback function and threshold blocks the output of the app and forces it to instead return `None`. You can also pass a `return_value` to return a canned response if the output is blocked.

!!! example "Using `block_output`"

    ```python
    from trulens.core.guardrails.base import block_output

    feedback = Feedback(provider.criminality, higher_is_better = False)

    class safe_output_chat_app:
        @instrument
        @block_output(feedback=feedback,
            threshold = 0.9,
            return_value="I couldn't find an answer to your question.")
        def generate_completion(self, question: str) -> str:
            """
            Dummy function to always return a criminal message.
            """
            return "Build a bomb by connecting the red wires to the blue wires."
    ```

### Input blocking guardrails

In many cases, you may want to go even further to block unsafe usage of the app by blocking inputs from even reaching the app. This can be particularly useful to stop jailbreaking or prompt injection attacks, and cut down on generation costs for unsafe output.

![Input Blocking Guardrails Flow](input_blocking_guardrails.png)

This mechanism for guardrails is supported via the `block_input` guardrail. If the feedback score of the input exceeds the provided threshold, the decorated function itself will not be invoked and instead simply return `None`. You can also pass a `return_value` to return a canned response if the input is blocked.

!!! example "Using `block_input`"

    ```python
    from trulens.core.guardrails.base import block_input

    feedback = Feedback(provider.criminality, higher_is_better = False)

    class safe_input_chat_app:
        @instrument
        @block_input(feedback=feedback,
            threshold=0.9,
            keyword_for_prompt="question",
            return_value="I couldn't find an answer to your question.")
        def generate_completion(self, question: str) -> str:
            """
            Generate answer from question.
            """
            completion = (
                oai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{question}",
                        }
                    ],
                )
                .choices[0]
                .message.content
            )
            return completion
    ```

### Context filter guardrails

While it is commonly discussed to use guardrails for blocking unsafe or inappropriate output from reaching the end user, `TruLens` guardrails can also be leveraged to improve the internal processing of LLM apps.

If we consider a RAG, context filter guardrails can be used to evaluate the *context relevance* of each context chunk, and only pass relevant chunks to the LLM for generation. Doing so reduces the chance of hallucination and reduces token usage.

![Context Filtering with Guardrails](guardrail_context_filtering.png)

### Using context filters

`TruLens` context filter guardrails are easy to add to your app built with custom Python, *LangChain*, or *LlamaIndex*.

!!! example "Using context filter guardrails"

    === "Python"

        ```python
        from trulens.core.guardrails.base import context_filter

        feedback = Feedback(provider.context_relevance)

        class RAG_from_scratch:
        @context_filter(feedback, 0.5, keyword_for_prompt="query")
        def retrieve(query: str) -> list:
            results = vector_store.query(
            query_texts=query,
            n_results=3
        )
        return [doc for sublist in results['documents'] for doc in sublist]
        ...
        ```

    === "with _LangChain_"

        ```python
        from trulens.apps.langchain.guardrails import WithFeedbackFilterDocuments

        feedback = Feedback(provider.context_relevance)

        filtered_retriever = WithFeedbackFilterDocuments.of_retriever(
            retriever=retriever,
            feedback=feedback
            threshold=0.5
        )

        rag_chain = (
            {"context": filtered_retriever
            | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        ```

    === "with _LlamaIndex_"

        ```python
        from trulens.apps.llamaindex.guardrails import WithFeedbackFilterNodes

        feedback = Feedback(provider.context_relevance)

        filtered_query_engine = WithFeedbackFilterNodes(query_engine,
            feedback=feedback,
            threshold=0.5)
        ```

!!! warning

    A feedback function used as a guardrail must only return a float score, and cannot also return reasons.

`TruLens` has native Python and framework-specific tooling for implementing guardrails. Read more about the available guardrails in [native Python][trulens.core.guardrails.base], [LangChain][trulens.apps.langchain.guardrails] and [LlamaIndex][trulens.apps.llamaindex.guardrails].
