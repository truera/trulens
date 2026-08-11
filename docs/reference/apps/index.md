# Apps

Apps derive from [AppDefinition][trulens.core.schema.app.AppDefinition] and
[App][trulens.core.app.App].

## 🦑 Core Apps

- [TruBasicApp][trulens.apps.basic.TruBasicApp]

- [TruApp][trulens.apps.app.TruApp]

- [TruVirtual][trulens.apps.virtual.TruVirtual]

## 📦 Optional Apps

- [TruChain][trulens.apps.langchain.TruChain] in package `trulens-apps-langchain`.

    ```bash
    pip install trulens-apps-langchain
    ```

- [TruLlama][trulens.apps.llamaindex.TruLlama] in package `trulens-apps-llamaindex`.

    ```bash
    pip install trulens-apps-llamaindex
    ```

- [TruGraph][trulens.apps.langgraph.TruGraph] in package `trulens-apps-langgraph`.
  Subclasses [TruChain][trulens.apps.langchain.TruChain], so LangGraph apps get
  LangChain instrumentation as well as graph node and edge spans.

    ```bash
    pip install trulens-apps-langgraph
    ```

- [TruRails][trulens.apps.nemo.TruRails] in package `trulens-apps-nemo`.

    !!! warning "Not installable alongside LangGraph"

        `nemoguardrails` requires `langchain<0.4`, while `trulens-apps-langgraph`
        requires `langchain>=1.0`. The two cannot coexist in one environment, so
        `trulens-apps-nemo` is not part of the default install. To use it, pin
        LangChain and skip the LangGraph package:

        ```bash
        pip install "langchain<1.0" trulens-apps-nemo
        ```

## 🧬 Prompt optimization

Not a recorder — this wraps a feedback function instead of an app.

- [TruGEPA][trulens.apps.gepa.TruGEPA] in package `trulens-apps-gepa`. Adapts a
  TruLens feedback function into a GEPA fitness function, and logs each candidate
  prompt as a [TruVirtual][trulens.apps.virtual.TruVirtual] record so the whole
  optimization run is traceable.

    ```bash
    pip install trulens-apps-gepa
    ```
