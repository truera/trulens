# 🤖 _AutoGen_ Integration

TruLens provides `TruAutoGen`, an integration with [_AutoGen_ (_AG2_)](https://github.com/ag2ai/ag2)
that records a multi-agent conversation as a span tree: who spoke, what they
were told, what they replied, and which tools they ran.

`TruAutoGen` offers:

* Agent turns as `AGENT` spans, carrying the agent's name, system message,
  message history, and reply
* Tool executions as `TOOL` spans nested under the agent that ran them
* Group chat speaker selection as `WORKFLOW_STEP` spans
* LLM calls as `GENERATION` spans, with costs from the configured provider
* Whole-conversation input and output on the record root

```bash
pip install trulens-apps-autogen
```

## Instrumenting AutoGen apps

Wrapping any one agent instruments the AutoGen classes themselves, so every
agent in the conversation is recorded — including agents created after the
recorder and the members of a group chat.

!!! example "Record a two-agent conversation"

    ```python
    from autogen import AssistantAgent, UserProxyAgent
    from trulens.apps.autogen import TruAutoGen

    assistant = AssistantAgent("assistant", llm_config=llm_config)
    user = UserProxyAgent("user", human_input_mode="NEVER", code_execution_config=False)

    tru_recorder = TruAutoGen(
        user,
        app_name="sky_explainer",
        app_version="v1",
        main_method=user.initiate_chat,
    )

    with tru_recorder as recording:
        result = user.initiate_chat(assistant, message="Why is the sky blue?", max_turns=2)
    ```

`main_method` marks where a record starts and ends. It is optional: without
it, the outermost instrumented AutoGen call inside the recording context starts
the record instead.

## Group chats

A group chat produces a nested tree — the manager's turn, then one speaker
selection and one agent reply per round, with any tool calls beneath the agent
that made them.

!!! example "Record a group chat"

    ```python
    from autogen import GroupChat, GroupChatManager
    from trulens.apps.autogen import TruAutoGen

    group_chat = GroupChat(
        agents=[researcher, writer],
        messages=[],
        max_round=6,
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    tru_recorder = TruAutoGen(
        user,
        app_name="research_team",
        app_version="v1",
        main_method=user.initiate_chat,
    )

    with tru_recorder as recording:
        user.initiate_chat(manager, message="Write a short post about Rayleigh scattering.")
    ```

## Evaluating individual agents

Because every agent turn is its own span, a metric can be pointed at agent
replies rather than at the conversation as a whole. Each `AGENT` span is
scored separately, so a weak agent shows up on its own rather than being
averaged into the final answer.

!!! example "Score every agent reply"

    ```python
    from trulens.core import Feedback
    from trulens.core.feedback.selector import Selector
    from trulens.otel.semconv.trace import SpanAttributes

    f_coherence = Feedback(
        provider.coherence_with_cot_reasons, name="Agent Coherence"
    ).on({
        "text": Selector(
            span_type=SpanAttributes.SpanType.AGENT,
            span_attribute=SpanAttributes.AGENT.OUTPUT_MESSAGE,
        ),
    })
    ```

The attributes available on an `AGENT` span are `AGENT.NAME`,
`AGENT.SYSTEM_MESSAGE`, `AGENT.DESCRIPTION`, `AGENT.INPUT_MESSAGES` (the
history the agent was given for that turn), `AGENT.INPUT_MESSAGE` and
`AGENT.INPUT_MESSAGE_ROLE` (the most recent message), and
`AGENT.OUTPUT_MESSAGE` (the reply). Speaker selection spans carry
`WORKFLOW.SPEAKER_SELECTION_METHOD`, `WORKFLOW.PARTICIPANT_AGENT_NAMES`, and
the selected agent as `WORKFLOW.AGENT_NAME`. Tool spans use the OpenTelemetry
GenAI convention: `gen_ai.tool.name`, `gen_ai.tool.call.arguments`, and
`gen_ai.tool.call.result`.

For a full walkthrough, see the
[Evaluate AutoGen Group Chat Quality](../../cookbook/frameworks/autogen/autogen_group_chat_quality.ipynb)
notebook.
