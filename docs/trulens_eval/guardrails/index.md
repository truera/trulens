# Guardrails

Guardrails play a crucial role in ensuring that only high quality output is produced by LLM apps. By setting guardrail thresholds based on feedback functions, we can directly leverage the same trusted evaluation metrics used for observability, *at inference time*.

## Typical guardrail usage

Typical guardrails *only* allow decisions based on the output, and have no impact on the intermediate steps of an LLM application.

![Standard Guardrails Flow](simple_guardrail_flow.png)

## _TruLens_ guardrails for internal steps

While it is commonly discussed to use guardrails for blocking unsafe or inappropriate output from reaching the end user, _TruLens_ guardrails can also be leveraged to improve the internal processing of LLM apps. 

If we consider a RAG, context filter guardrails can be used to evaluate the *context relevance* of each context chunk, and only pass relevant chunks to the LLM for generation. Doing so reduces the chance of hallucination and reduces token usage.

![Context Filtering with Guardrails](guardrail_context_filtering.png)

## Using _TruLens_ guardrails

_TruLens_ context filter guardrails are easy to add to your _Langchain_ or _Llama-Index_ app.

!!! example "Using context filter guardrails"

    === "in _Langchain_"

        ```python
        from trulens_eval.guardrails.langchain import WithFeedbackFilterDocuments

        f_context_relevance = (
            Feedback(provider.context_relevance)
            .on_input()
            .on(context)
            .aggregate(np.mean)
        )

        filtered_retriever = WithFeedbackFilterDocuments.of_retriever(
            retriever=retriever,
            feedback=f_context_relevance,
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

    === "in _Llama-Index_"

        ```python
        from trulens_eval.guardrails.llama import WithFeedbackFilterNodes

        f_context_relevance = (
            Feedback(provider.context_relevance)
            .on_input()
            .on(context)
            .aggregate(np.mean)
        )

        filtered_query_engine = WithFeedbackFilterNodes(query_engine,
            feedback=f_context_relevance,
            threshold=0.5)
        ```

!!! warning

    Feedback function used as a guardrail must only return a float score, and cannot also return reasons.

TruLens has framework-specific tooling for implementing guardrails. Read more about the availble guardrails in [Langchain]() and [Llama-Index]().