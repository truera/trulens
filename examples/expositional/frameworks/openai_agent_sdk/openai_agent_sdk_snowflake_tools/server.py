"""FastAPI backend with trace_with_run for live TruLens tracing from the React chat UI."""

import asyncio
import datetime
import json
import logging
import threading
import time

from agents.stream_events import RunItemStreamEvent, RawResponsesStreamEvent
from agents.items import ToolCallItem, ToolCallOutputItem, MessageOutputItem
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from trulens.core.app import trace_with_run
from trulens.core.run import RunConfig, RunStatus

from src.eval.metrics import SERVERSIDE_METRICS
from src.observability.trulens_setup import setup_observability

logger = logging.getLogger(__name__)

agent_app, tru_app, session, sf_connector, all_metrics = setup_observability()

app = FastAPI(title="Support Intelligence Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    run_name: str | None = None


class ChatResponse(BaseModel):
    response: str
    run_name: str


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _compute_metrics_bg(run_name: str):
    try:
        run_config = RunConfig(
            run_name=run_name,
            dataset_name=f"live_{run_name}",
            source_type="DATAFRAME",
            dataset_spec={},
            llm_judge_name="openai-gpt-5.1",
        )
        run = tru_app.add_run(run_config=run_config)

        for _ in range(20):
            status = run.get_status()
            if status in {
                RunStatus.INVOCATION_COMPLETED,
                RunStatus.INVOCATION_PARTIALLY_COMPLETED,
            }:
                break
            time.sleep(3)

        metrics = all_metrics + SERVERSIDE_METRICS
        run.compute_metrics(metrics=metrics)
        logger.info(f"[{run_name}] Metrics computed successfully")
    except Exception as e:
        logger.warning(f"[{run_name}] Background metrics failed: {e}")


@app.post("/api/chat")
def chat(req: ChatRequest):
    run_name = req.run_name or f"PROD_MONITOR_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    @trace_with_run(app=tru_app, run_name=run_name)
    def process_message(message: str):
        return agent_app.ask(message)

    result = process_message(req.message)

    threading.Thread(
        target=_compute_metrics_bg, args=(run_name,), daemon=True
    ).start()

    return ChatResponse(response=result, run_name=run_name)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    run_name = req.run_name or f"PROD_MONITOR_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    async def generate():
        yield _sse("status", {"message": "Thinking...", "run_name": run_name})

        event_queue: asyncio.Queue = asyncio.Queue()

        @trace_with_run(app=tru_app, run_name=run_name)
        async def run_agent(message: str):
            return await agent_app.ask_streamed(message, event_queue)

        agent_task = asyncio.create_task(run_agent(req.message))

        while True:
            event = await event_queue.get()
            if event is None:
                break
            if isinstance(event, RunItemStreamEvent):
                item = event.item
                if isinstance(item, ToolCallItem):
                    raw = item.raw_item
                    name = getattr(getattr(raw, "function", None), "name", None) or getattr(raw, "name", "tool")
                    yield _sse("tool_start", {"tool": name})
                elif isinstance(item, ToolCallOutputItem):
                    raw = item.raw_item
                    name = getattr(raw, "call_id", "tool")
                    output_text = getattr(item, "output", "")
                    if isinstance(output_text, str) and len(output_text) > 200:
                        output_text = output_text[:200] + "..."
                    yield _sse("tool_done", {"tool": name, "preview": str(output_text)[:200]})
                elif isinstance(item, MessageOutputItem):
                    text = ""
                    raw = item.raw_item
                    for content_part in getattr(raw, "content", []):
                        if hasattr(content_part, "text"):
                            text += content_part.text
                    if text:
                        yield _sse("response", {"text": text})
            elif isinstance(event, RawResponsesStreamEvent):
                data = event.data
                if hasattr(data, "type") and data.type == "response.output_text.delta":
                    delta = getattr(data, "delta", "")
                    if delta:
                        yield _sse("delta", {"text": delta})

        final = await agent_task
        yield _sse("done", {"text": final or "", "run_name": run_name})

    async def generate_with_metrics():
        async for chunk in generate():
            yield chunk
        threading.Thread(
            target=_compute_metrics_bg, args=(run_name,), daemon=True
        ).start()

    return StreamingResponse(generate_with_metrics(), media_type="text/event-stream")


@app.get("/api/health")
def health():
    return {"status": "ok"}
