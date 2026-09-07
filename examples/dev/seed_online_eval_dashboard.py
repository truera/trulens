from __future__ import annotations

import argparse
import datetime
import hashlib
import pathlib
import random
from typing import Optional

from trulens.core import session as core_session
from trulens.core.schema import app as app_schema
from trulens.core.schema import event as event_schema
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes

APP_PROFILES = {
    "support-rag": "improving",
    "commerce-agent": "regressing",
    "policy-assistant": "stable",
}
VERSIONS = ("v1", "v2", "v3")
METRICS = ("groundedness", "answer_relevance", "context_relevance")

APP_CONTENT = {
    "support-rag": (
        (
            "I cannot sign in after resetting my password.",
            "Confirm the reset email is the newest one, then clear the old session and sign in again. If the lock persists, I can help open an access ticket.",
            "KB-104: Password reset links expire after 30 minutes. Existing sessions may need to be cleared after a reset.",
        ),
        (
            "The reset worked, but my account now says it is locked.",
            "The lockout normally clears after 15 minutes. If it does not, use the account recovery form and include the lockout timestamp.",
            "KB-118: Five failed sign-ins trigger a 15-minute lockout. Support can manually unlock verified users.",
        ),
        (
            "Where can I see the status of that access ticket?",
            "Open Support Center, choose My Requests, and select the access ticket. Status changes and agent notes appear in its activity feed.",
            "KB-203: Customers can track requests under Support Center > My Requests and receive email notifications on updates.",
        ),
        (
            "Can I add my manager to the ticket notifications?",
            "Yes. Open the ticket, choose Share, and add your manager as a follower. They will receive future status notifications.",
            "KB-207: Ticket owners can add followers from the Share menu. Followers receive updates but cannot edit the request.",
        ),
    ),
    "commerce-agent": (
        (
            "Do you have the trail shoes in blue, size 10?",
            "The Ridge Runner is available in blue, size 10, with 14 units currently in stock.",
            "catalog_search",
        ),
        (
            "Add one pair and apply my loyalty discount.",
            "I added one pair to your cart and applied the 10% Trail Club discount. The discounted subtotal is $116.10.",
            "cart_update",
        ),
        (
            "Will it arrive in Portland by Friday?",
            "Standard delivery is estimated for Friday to Portland 97205. Expedited delivery is available for an additional $12.",
            "shipping_quote",
        ),
        (
            "Use standard shipping and place the order.",
            "Your order has been placed with standard shipping. The confirmation number is TR-48217.",
            "checkout",
        ),
    ),
    "policy-assistant": (
        (
            "Can contractors access customer production data?",
            "Only approved contractors with a documented business need may receive time-bound access, and the data owner must approve it.",
            "Access Control Policy §4.2: Contractor access requires owner approval, least privilege, and an expiration date.",
        ),
        (
            "How long can that access remain active?",
            "Contractor access must expire within 90 days and be reviewed before renewal. Shorter engagement dates should be used when applicable.",
            "Access Control Policy §4.4: External-user grants expire after at most 90 days and require recertification for renewal.",
        ),
        (
            "Is MFA required for the contractor account?",
            "Yes. MFA is mandatory for every external identity before production access is activated.",
            "Identity Standard §3.1: MFA is required for workforce and external identities accessing production systems.",
        ),
        (
            "What evidence should the approver retain?",
            "Retain the request, business justification, data-owner approval, granted role, and expiration date in the access review system.",
            "Access Control Policy §6.2: Approval evidence must include requester, justification, owner, entitlements, and expiration.",
        ),
    ),
}


def _event(
    event_id: str,
    app_name: str,
    app_version: str,
    app_id: str,
    record_id: str,
    span_type: str,
    start: datetime.datetime,
    attributes: dict,
    duration_seconds: float = 0.75,
    parent_id: str = "",
    trace_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    name: Optional[str] = None,
) -> event_schema.Event:
    record_attributes = {
        SpanAttributes.SPAN_TYPE: span_type,
        SpanAttributes.RECORD_ID: record_id,
        **{
            key: value for key, value in attributes.items() if value is not None
        },
    }
    if conversation_id:
        record_attributes[SpanAttributes.CONVERSATION_ID] = conversation_id
    return event_schema.Event(
        event_id=event_id,
        record={"name": name or f"validation.{span_type}"},
        record_attributes=record_attributes,
        record_type=event_schema.EventRecordType.SPAN,
        resource_attributes={
            ResourceAttributes.APP_NAME: app_name,
            ResourceAttributes.APP_VERSION: app_version,
            ResourceAttributes.APP_ID: app_id,
        },
        start_timestamp=start,
        timestamp=start + datetime.timedelta(seconds=duration_seconds),
        trace={
            "trace_id": trace_id or record_id,
            "parent_id": parent_id,
            "span_id": event_id,
        },
    )


def _conversation_layout(
    records_per_version: int, rng
) -> dict[int, tuple[str, int, int]]:
    conversational_count = int(records_per_version * 0.75)
    layout = {}
    record_index = 0
    conversation_index = 0
    while record_index < conversational_count:
        turns = min(rng.randint(3, 8), conversational_count - record_index)
        if turns < 3:
            break
        conversation_id = f"conversation-{conversation_index:03d}"
        for turn in range(turns):
            layout[record_index + turn] = (
                conversation_id,
                turn,
                record_index,
            )
        record_index += turns
        conversation_index += 1
    return layout


def _content(app_name: str, record_index: int, turn: Optional[int]):
    cases = APP_CONTENT[app_name]
    case_index = turn if turn is not None else record_index
    question, answer, context_or_tool = cases[case_index % len(cases)]
    if record_index % 97 == 0:
        answer = "I could not complete that request because an upstream dependency timed out. Please retry."
    return question, answer, context_or_tool


def _app_trace_events(
    app_name: str,
    app_version: str,
    app_id: str,
    record_id: str,
    prefix: str,
    start: datetime.datetime,
    latency: float,
    question: str,
    answer: str,
    context_or_tool: str,
    conversation_id: Optional[str],
    record_index: int,
    rng,
):
    root_id = f"{prefix}-record"
    workflow_id = f"{prefix}-workflow"
    operation_id = f"{prefix}-operation"
    generation_id = f"{prefix}-generation"
    trace_id = hashlib.sha256(f"trace:{record_id}".encode()).hexdigest()[:32]
    prompt_tokens = 220 + rng.randint(0, 420)
    completion_tokens = 70 + rng.randint(0, 210)
    app_cost = round(0.002 + rng.random() * 0.018, 5)
    failed = record_index % 97 == 0
    root_attributes = {
        SpanAttributes.RECORD_ROOT.INPUT: question,
    }
    if failed:
        root_attributes[SpanAttributes.RECORD_ROOT.ERROR] = (
            "UpstreamDependencyTimeout: operation exceeded 10 seconds"
        )
    else:
        root_attributes[SpanAttributes.RECORD_ROOT.OUTPUT] = answer
    events = [
        _event(
            root_id,
            app_name,
            app_version,
            app_id,
            record_id,
            SpanAttributes.SpanType.RECORD_ROOT.value,
            start,
            root_attributes,
            duration_seconds=latency,
            trace_id=trace_id,
            conversation_id=conversation_id,
            name=f"{app_name}.invoke",
        ),
        _event(
            workflow_id,
            app_name,
            app_version,
            app_id,
            record_id,
            SpanAttributes.SpanType.AGENT.value,
            start + datetime.timedelta(seconds=latency * 0.04),
            {
                SpanAttributes.WORKFLOW.AGENT_NAME: app_name,
                SpanAttributes.WORKFLOW.INPUT_EVENT: question,
                SpanAttributes.WORKFLOW.OUTPUT_EVENT: answer,
            },
            duration_seconds=latency * 0.91,
            parent_id=root_id,
            trace_id=trace_id,
            conversation_id=conversation_id,
            name=f"{app_name}.agent",
        ),
    ]
    if app_name == "commerce-agent":
        events.append(
            _event(
                operation_id,
                app_name,
                app_version,
                app_id,
                record_id,
                SpanAttributes.SpanType.TOOL.value,
                start + datetime.timedelta(seconds=latency * 0.12),
                {
                    GenAIAttributes.TOOL.NAME: context_or_tool,
                    GenAIAttributes.TOOL.CALL_ARGUMENTS: (
                        f'{{"query":"{question}","customer_id":"demo-42"}}'
                    ),
                    GenAIAttributes.TOOL.CALL_RESULT: (
                        '{"status":"timeout"}'
                        if failed
                        else f'{{"status":"ok","result":"{answer}"}}'
                    ),
                    SpanAttributes.CALL.FUNCTION: context_or_tool,
                    SpanAttributes.CALL.ERROR: (
                        "TimeoutError: commerce service did not respond"
                        if failed
                        else None
                    ),
                },
                duration_seconds=latency * 0.35,
                parent_id=workflow_id,
                trace_id=trace_id,
                conversation_id=conversation_id,
                name=f"tool.{context_or_tool}",
            )
        )
    else:
        events.append(
            _event(
                operation_id,
                app_name,
                app_version,
                app_id,
                record_id,
                SpanAttributes.SpanType.RETRIEVAL.value,
                start + datetime.timedelta(seconds=latency * 0.12),
                {
                    SpanAttributes.RETRIEVAL.QUERY_TEXT: question,
                    SpanAttributes.RETRIEVAL.NUM_CONTEXTS: 3,
                    SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [
                        context_or_tool,
                        f"Related guidance for case {record_index % 17}.",
                        "Escalate when the documented workflow cannot resolve the request.",
                    ],
                    GenAIAttributes.RETRIEVAL.QUERY_TEXT: question,
                    GenAIAttributes.RETRIEVAL.DOCUMENTS: [context_or_tool],
                },
                duration_seconds=latency * 0.28,
                parent_id=workflow_id,
                trace_id=trace_id,
                conversation_id=conversation_id,
                name=f"{app_name}.retrieve",
            )
        )
    events.append(
        _event(
            generation_id,
            app_name,
            app_version,
            app_id,
            record_id,
            SpanAttributes.SpanType.GENERATION.value,
            start + datetime.timedelta(seconds=latency * 0.48),
            {
                GenAIAttributes.OPERATION.NAME: "chat",
                GenAIAttributes.REQUEST.MODEL: "snowflake-arctic",
                GenAIAttributes.REQUEST.TEMPERATURE: 0.2,
                GenAIAttributes.USAGE.INPUT_TOKENS: prompt_tokens,
                GenAIAttributes.USAGE.OUTPUT_TOKENS: completion_tokens,
                GenAIAttributes.SYSTEM.NAME: "snowflake",
                SpanAttributes.CALL.FUNCTION: "generate_response",
                SpanAttributes.CALL.RETURN: answer if not failed else None,
                SpanAttributes.CALL.ERROR: (
                    "UpstreamDependencyTimeout" if failed else None
                ),
                SpanAttributes.COST.NUM_TOKENS: prompt_tokens
                + completion_tokens,
                SpanAttributes.COST.NUM_PROMPT_TOKENS: prompt_tokens,
                SpanAttributes.COST.NUM_COMPLETION_TOKENS: completion_tokens,
                SpanAttributes.COST.COST: app_cost,
                SpanAttributes.COST.CURRENCY: "USD",
                SpanAttributes.COST.MODEL: "snowflake-arctic",
            },
            duration_seconds=latency * 0.47,
            parent_id=workflow_id,
            trace_id=trace_id,
            conversation_id=conversation_id,
            name="chat snowflake-arctic",
        )
    )
    return events


def _score(profile: str, version_index: int, day: int, metric_index: int, rng):
    version_offset = version_index * 0.06
    if profile == "improving":
        trend = day * 0.008
    elif profile == "regressing":
        trend = -day * 0.007
    else:
        trend = 0
    value = 0.58 + version_offset + trend + metric_index * 0.025
    return min(0.98, max(0.02, value + rng.gauss(0, 0.08)))


def _latency(
    profile: str, version_index: int, day: int, record_index: int, rng
) -> float:
    version_factor = (1.0, 0.82, 0.66)[version_index]
    if profile == "improving":
        day_factor = 1.15 - day * 0.012
    elif profile == "regressing":
        day_factor = 0.85 + day * 0.018
    else:
        day_factor = 1.0 + 0.12 * ((day % 7) / 6)
    latency = 0.55 * version_factor * day_factor
    latency *= rng.lognormvariate(0, 0.38)
    if record_index % 47 == 0:
        latency *= 4.5
    elif record_index % 19 == 0:
        latency *= 2.2
    return max(0.08, round(latency, 4))


def seed(database_path: pathlib.Path, records_per_version: int) -> None:
    database_path.unlink(missing_ok=True)
    session = core_session.TruSession(database_url=f"sqlite:///{database_path}")
    session.reset_database()
    db = session.connector.db
    rng = random.Random(2619)
    start_date = datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc)
    events = []

    for app_name, profile in APP_PROFILES.items():
        for version_index, app_version in enumerate(VERSIONS):
            app = app_schema.AppDefinition(
                app_name=app_name,
                app_version=app_version,
                root_class={
                    "name": "ValidationApp",
                    "module": {"module_name": "validation"},
                },
                app={},
            )
            session.add_app(app)
            conversation_layout = _conversation_layout(records_per_version, rng)
            for record_index in range(records_per_version):
                conversation = conversation_layout.get(record_index)
                if conversation:
                    raw_conversation_id, turn, conversation_start = conversation
                    conversation_id = (
                        f"{app_name}-{app_version}-{raw_conversation_id}"
                    )
                    day = conversation_start % 30
                    hour = (conversation_start * 7) % 24
                    start = start_date + datetime.timedelta(
                        days=day,
                        hours=hour,
                        minutes=turn * 4,
                    )
                else:
                    conversation_id = None
                    turn = None
                    day = record_index % 30
                    hour = (record_index * 7) % 24
                    start = start_date + datetime.timedelta(
                        days=day, hours=hour
                    )
                record_id = hashlib.sha256(
                    f"{app.app_id}:{record_index}".encode()
                ).hexdigest()[:24]
                prefix = f"{app.app_id}-{record_index}"
                question, answer, context_or_tool = _content(
                    app_name, record_index, turn
                )
                latency = _latency(
                    profile,
                    version_index,
                    day,
                    record_index,
                    rng,
                )
                events.extend(
                    _app_trace_events(
                        app_name,
                        app_version,
                        app.app_id,
                        record_id,
                        prefix,
                        start,
                        latency,
                        question,
                        answer,
                        context_or_tool,
                        conversation_id,
                        record_index,
                        rng,
                    )
                )

                rate = (1.0, 0.25, 0.1)[version_index]
                if (
                    app_name == "support-rag"
                    and app_version == "v3"
                    and day >= 15
                ):
                    rate = 0.05
                hash_value = int(record_id[:8], 16) / 0xFFFFFFFF
                reason = "evaluated" if hash_value < rate else "not_sampled"
                if record_index % 127 == 0:
                    reason = "throttled"
                elif record_index % 211 == 0:
                    reason = "over_budget"
                if not (app_name == "policy-assistant" and app_version == "v1"):
                    events.append(
                        _event(
                            f"{prefix}-decision",
                            app_name,
                            app_version,
                            app.app_id,
                            record_id,
                            SpanAttributes.SpanType.EVAL_DECISION.value,
                            start + datetime.timedelta(milliseconds=10),
                            {
                                SpanAttributes.EVAL_DECISION.SAMPLE_RATE: rate,
                                SpanAttributes.EVAL_DECISION.EVAL_DECISION_REASON: reason,
                            },
                            parent_id=f"{prefix}-record",
                            trace_id=hashlib.sha256(
                                f"trace:{record_id}".encode()
                            ).hexdigest()[:32],
                            conversation_id=conversation_id,
                        )
                    )

                if reason != "evaluated":
                    continue
                for metric_index, metric_name in enumerate(METRICS):
                    events.append(
                        _event(
                            f"{prefix}-eval-{metric_name}",
                            app_name,
                            app_version,
                            app.app_id,
                            record_id,
                            SpanAttributes.SpanType.EVAL_ROOT.value,
                            start + datetime.timedelta(milliseconds=20),
                            {
                                SpanAttributes.EVAL_ROOT.METRIC_NAME: metric_name,
                                SpanAttributes.EVAL_ROOT.SCORE: _score(
                                    profile,
                                    version_index,
                                    day,
                                    metric_index,
                                    rng,
                                ),
                                SpanAttributes.COST.COST: round(
                                    0.0004 + rng.random() * 0.002, 6
                                ),
                                SpanAttributes.COST.CURRENCY: "USD",
                            },
                            trace_id=hashlib.sha256(
                                f"eval:{record_id}:{metric_name}".encode()
                            ).hexdigest()[:32],
                            conversation_id=conversation_id,
                            name=f"evaluate.{metric_name}",
                        )
                    )
                if len(events) >= 1000:
                    db.insert_events(events)
                    events.clear()
    if events:
        db.insert_events(events)
    print(f"Seeded {len(APP_PROFILES)} apps, {len(APP_PROFILES) * 3} versions")
    print(f"Records: {len(APP_PROFILES) * 3 * records_per_version}")
    print(f"Database: {database_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=pathlib.Path, required=True)
    parser.add_argument("--records-per-version", type=int, default=500)
    args = parser.parse_args()
    seed(args.database.expanduser().resolve(), args.records_per_version)


if __name__ == "__main__":
    main()
