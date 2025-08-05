# 🦜️ _LangGraph_ Integration

TruLens provides TruGraph, a deep integration with _LangGraph_ that allows you to
inspect and evaluate the internals of your _LangGraph_-built applications.

TruGraph offers:

* Automatic detection of LangGraph applications
* Combined instrumentation of both LangChain and LangGraph components
* Multi-agent evaluation capabilities
* Automatic @task instrumentation with intelligent attribute extraction

## Instrumenting LangGraph apps

To demonstrate usage, we'll create a basic multi-agent workflow with a researcher and a writer.

First, this requires loading data into a vector store.

!!! example "Create an agent with LangGraph"

    ```python
    def research_agent(state):
        """Agent that performs research on a topic."""
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                query = last_message.content
            else:
                query = str(last_message)

            # Simulate research (in a real app, this would call external APIs)
            research_results = f"Research findings for '{query}': This is a comprehensive analysis of the topic."
            return {"messages": [AIMessage(content=research_results)]}

        return {"messages": [AIMessage(content="No research query provided")]}

    def writer_agent(state):
        """Agent that writes articles based on research."""
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                research_content = last_message.content
            else:
                research_content = str(last_message)

            # Simulate article writing
            article = f"Article: Based on the research - {research_content[:100]}..."
            return {"messages": [AIMessage(content=article)]}

        return {"messages": [AIMessage(content="No research content provided")]}

    # Create the workflow
    workflow = StateGraph(MessagesState)
    workflow.add_node("researcher", research_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)
    workflow.set_entry_point("researcher")

    # Compile the graph
    graph = workflow.compile()

    print("✅ Multi-agent workflow created successfully!")
    print(f"Graph type: {type(graph)}")
    print(f"Graph module: {graph.__module__}")

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }
    ```

To instrument the graph, all that's required is to wrap it using TruGraph.

!!! example "Instrument with `TruGraph`"

    ```python
    from trulens.apps.langgraph import TruGraph

    tru_recorder = TruGraph(graph,
        app_name="tru_simple_graph",
        app_version="v1.0"
        )
    ```

## Auto-Detection of apps using LangGraph `@task` decorator

One of the key features of TruGraph is its ability to automatically detect and instrument functions decorated with LangGraph's @task decorator. This means you can use standard LangGraph patterns without any additional instrumentation code.

How it works:

1. Automatic Detection: TruGraph automatically scans for functions decorated with @task
2. Smart Attribute Extraction: It intelligently extracts information from function arguments:
   * Handles BaseChatModel and BaseModel objects
   * Extracts data from dataclasses and Pydantic models
   * Skips non-serializable objects like LLM pools
   * Captures return values and exceptions
3. Seamless Integration: No additional decorators or code changes required

In the example below, `my_agent_function` is automatically instrumented. No manual setup required!

!!! example

    ```python

    from langgraph.func import task

    @task  # This is automatically detected and instrumented by TruGraph
    def my_agent_function(state, config):
        # Your agent logic here
        return updated_state
    ```

### Instrumentation of custom classes

Beyond instrumenting explicit _LangGraph_ classes, `TruGraph` can also be used to instrument custom classes leveraging the `@task` decorator. Consider a more complete example using same researcher/writer multi-agent system we built before.

In this example, the `write_essay` method is automatically instrumented by `TruGraph`.

In addition, we manually instrument the `preprocess` with _TruLens_ `@instrument`.

!!! example

    ```python
    import pandas as pd
    from trulens.apps.langgraph import TruGraph
    from trulens.core.otel.instrument import instrument

    from langgraph.func import entrypoint, task
    from langgraph.types import interrupt
    from langgraph.checkpoint.memory import MemorySaver

    @instrument()
    def preprocess_input(topic: str) -> str:
        """Custom preprocessing step."""
        return f"Preprocessed {topic}"

    @task
    def write_essay(topic: str) -> str:
        """Write an essay about the given topic."""
        return f"An essay about topic: {topic}"

    @entrypoint(checkpointer=MemorySaver())
    def workflow(topic: str) -> dict:
        """A simple workflow that writes an essay and asks for a review."""
        essay = write_essay("cat").result()
        is_approved = interrupt({
            # Any json-serializable payload provided to interrupt as argument.
            # It will be surfaced on the client side as an Interrupt when streaming data
            # from the workflow.
            "essay": essay, # The essay we want reviewed.
            # We can add any additional information that we need.
            # For example, introduce a key called "action" with some instructions.
            "action": "Please approve/reject the essay",
        })

        return {
            "essay": essay, # The essay that was generated
            "is_approved": is_approved, # Response from HIL
        }

    class ComplexRAGAgent:
        def __init__(self):
            self.workflow = workflow
        def run(self, topic: str) -> dict:
            return self.workflow.invoke(topic)


    complex_agent = ComplexRAGAgent()

    tru_graph_complex_agent = TruGraph(complex_agent,
       app_name="essay_writer",
       app_version="base")

    with tru_graph_complex_agent as app:
        complex_agent.run("cat")
    ```

By combining auto-instrumentation of tasks and manual instrumentation in custom classes, you can capture the full execution flow across custom orchestration logic and LangGraph workflows. Your non-LangGraph steps are now included in traces, and you can evaluate end-to-end performance without blindspots.
