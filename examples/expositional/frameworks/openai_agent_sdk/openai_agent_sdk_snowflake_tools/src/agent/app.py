import asyncio
import json

from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    set_default_openai_api,
    set_tracing_disabled,
)
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from src.services.config import SNOWFLAKE_ACCOUNT_URL, SNOWFLAKE_PAT
from src.agent.tools import query_ticket_metrics, search_knowledge_base

set_tracing_disabled(True)
set_default_openai_api("chat_completions")


class _SnowflakeChatCompletions:
    def __init__(self, inner):
        self._inner = inner

    @staticmethod
    def _generation_attributes(ret, exception, *args, **kwargs):
        attrs = {}
        messages = kwargs.get("messages", [])
        if messages:
            attrs["ai.observability.generation.input_messages"] = json.dumps(
                [
                    {"role": getattr(m, "role", str(m.get("role", "") if isinstance(m, dict) else "")),
                     "content": getattr(m, "content", str(m.get("content", "") if isinstance(m, dict) else ""))[:500]}
                    for m in messages
                ],
                default=str,
            )
        if ret is None:
            return attrs
        attrs["ai.observability.generation.model"] = getattr(ret, "model", "") or kwargs.get("model", "")
        choices = getattr(ret, "choices", []) or []
        if choices:
            choice = choices[0]
            attrs["ai.observability.generation.finish_reason"] = getattr(choice, "finish_reason", "") or ""
            msg = getattr(choice, "message", None)
            if msg:
                attrs["ai.observability.generation.content"] = getattr(msg, "content", "") or ""
                tc = getattr(msg, "tool_calls", None)
                if tc:
                    attrs["ai.observability.generation.tool_calls"] = json.dumps(
                        [
                            {"id": getattr(t, "id", ""),
                             "type": getattr(t, "type", ""),
                             "function": {"name": getattr(getattr(t, "function", None), "name", ""),
                                          "arguments": getattr(getattr(t, "function", None), "arguments", "")}}
                            for t in tc
                        ],
                        default=str,
                    )
        usage = getattr(ret, "usage", None)
        if usage:
            attrs["ai.observability.generation.prompt_tokens"] = str(getattr(usage, "prompt_tokens", 0) or 0)
            attrs["ai.observability.generation.completion_tokens"] = str(getattr(usage, "completion_tokens", 0) or 0)
            attrs["ai.observability.generation.total_tokens"] = str(getattr(usage, "total_tokens", 0) or 0)
        return attrs

    @instrument(
        span_type=SpanAttributes.SpanType.GENERATION,
        attributes=_generation_attributes.__func__,
    )
    async def create(self, **kwargs):
        kwargs.pop("max_tokens", None)
        resp = await self._inner.create(**kwargs)
        for choice in getattr(resp, "choices", []):
            msg = getattr(choice, "message", None)
            if msg and getattr(msg, "audio", None):
                msg.audio = None
            if msg and getattr(msg, "function_call", None):
                fc = msg.function_call
                if not getattr(fc, "name", None) and not getattr(fc, "arguments", None):
                    msg.function_call = None
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _SnowflakeChat:
    def __init__(self, inner):
        self._inner = inner
        self.completions = _SnowflakeChatCompletions(inner.completions)

    def __getattr__(self, name):
        if name == "completions":
            return self.completions
        return getattr(self._inner, name)


class SnowflakeAsyncOpenAI(AsyncOpenAI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sf_chat = _SnowflakeChat(super().chat)

    @property
    def chat(self):
        return self._sf_chat


_snowflake_openai_client = SnowflakeAsyncOpenAI(
    api_key=SNOWFLAKE_PAT,
    base_url=f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/cortex/v1",
)

_model = OpenAIChatCompletionsModel(
    model="openai-gpt-5.1",
    openai_client=_snowflake_openai_client,
)

support_agent = Agent(
    name="Support Cloud Agent",
    instructions=(
        "You are a support intelligence assistant. You help answer questions "
        "about support ticket metrics and knowledge base articles.\n\n"
        "TOOL SELECTION:\n"
        "- Use query_ticket_metrics for questions about ticket counts, "
        "resolution times, CSAT scores, agent performance, priorities, "
        "or any quantitative ticket data.\n"
        "- Use search_knowledge_base for how-to questions, "
        "troubleshooting guidance, or information about features, billing, "
        "or account management.\n\n"
        "IMPORTANT RULES:\n"
        "- Base your answer ONLY on the specific information returned by tools. "
        "Do NOT add information that was not in the tool results.\n"
        "- If retrieved context does not directly address the question, "
        "say so rather than guessing.\n"
        "- Be concise and precise. Only include details that are relevant "
        "to the user's specific question.\n"
        "- If a tool returns an error, explain the issue and suggest alternatives."
    ),
    tools=[query_ticket_metrics, search_knowledge_base],
    model=_model,
)


class AgentApp:
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "question",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def ask(self, question: str) -> str:
        result = Runner.run_sync(support_agent, question)
        return result.final_output

    async def ask_streamed(self, question: str, event_queue: asyncio.Queue) -> str:
        result = Runner.run_streamed(support_agent, question)
        async for event in result.stream_events():
            await event_queue.put(event)
        await event_queue.put(None)
        return result.final_output
