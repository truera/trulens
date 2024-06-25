from copy import deepcopy
from datetime import datetime
import json
from typing import List, Literal, Optional

import pandas as pd
from pydantic import BaseModel
import streamlit as st
from streamlit_pills import pills

from trulens_eval.schema.feedback import FeedbackCall
from trulens_eval.tru_custom_app import TruCustomApp


class ModelConfig(BaseModel):
    model: str = "Snowflake Arctic"
    temperature: float = 0.7
    top_p: float = 1.0
    max_new_tokens: int = 1024
    system_prompt: str = ""
    use_rag: bool = True
    retriever: Optional[str] = "Cortex Search"
    retrieval_filter: Optional[float] = 0.5
    filter_feedback_function: Optional[str] = "Context Relevance (LLM-as-Judge)"
    trulens_recorder: Optional[TruCustomApp] = None


class FeedbackDisplay(BaseModel):
    score: float = 0
    calls: list[FeedbackCall]
    icon: str


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    feedbacks: dict[str, FeedbackDisplay] = {}
    sources: set[str] = set()


class ConversationFeedback(BaseModel):
    category: str
    custom_category: str = ""
    quality_score: float
    comments: str = ""
    flagged: bool = False
    flagged_comments: str = ""


class Conversation:
    messages: List[Message] = []
    model_config: ModelConfig = None
    feedback: ConversationFeedback = None
    has_error: bool = None

    def __init__(self):
        self.reset_messages()
        self.model_config: ModelConfig = ModelConfig()
        self.feedback: ConversationFeedback = None

    def add_message(self, message: Message, container=st, render=True):
        self.messages.append(message)
        if render:
            self.render_message(message, container, key=str(len(self.messages)))

    def reset_messages(self):
        self.messages: List[Message] = []

    def render_all(self, container=st):
        for idx, message in enumerate(self.messages):
            self.render_message(message, container, key=str(idx))

    @st.experimental_fragment(run_every=2)
    def render_feedbacks(self, message: Message, key: str):
        feedbacks: dict[str, FeedbackDisplay] = message.feedbacks
        if len(feedbacks) == 0:
            return

        feedback_cols = list(feedbacks.keys())
        icons = list(map(lambda fcol: feedbacks[fcol].icon, feedback_cols))

        st.write('**Feedback functions**')
        selected_fcol = pills(
            "Feedback functions",
            feedback_cols,
            index=None,
            format_func=lambda fcol: f"{fcol} {feedbacks[fcol].score:.4f}",
            label_visibility=
            "collapsed",  # Hiding because we can't format the label here.
            icons=icons,
            key=
            f"{key}_{len(feedbacks)}"  # Important! Otherwise streamlit sometimes lazily skips update even with st.exprimental_fragment
        )

        # Extract the arguments + meta from the feedback call into a dict
        def extract_call(fcall: FeedbackCall):
            ret = {}
            fcall_dump = fcall.model_dump()

            if 'args' in fcall_dump:
                for arg in fcall_dump['args'].keys():
                    ret[arg] = fcall_dump['args'][arg]

            ret['result'] = fcall_dump['ret']

            if 'meta' in fcall_dump:
                for met in fcall_dump['meta'].keys():
                    ret[met] = fcall_dump['meta'][met]

            return ret

        if selected_fcol != None:
            calls: list[FeedbackCall] = feedbacks[selected_fcol].calls
            calls_dict = list(map(lambda fcall: extract_call(fcall), calls))
            st.dataframe(
                pd.DataFrame.from_records(calls_dict),
                use_container_width=True,
                hide_index=True
            )

        if len(message.sources) > 0:
            with st.expander(f'**{len(message.sources)} sources used**'):
                st.dataframe(
                    pd.DataFrame(
                        list(message.sources), columns=['Source text']
                    ),
                    use_container_width=True,
                    hide_index=True
                )

    def render_message(
        self, message: Message, container=st, key=str(datetime.now())
    ):
        with container.chat_message(message.role):
            st.write(message.content)

            self.render_feedbacks(message, key)

    def messages_to_text(self, truncate=True):
        msgs = []
        for m in self.messages[1:]:
            if len(m.content) < 35 or not truncate:
                txt = m.content
            else:
                txt = m.content[0:35] + "..."
            msgs.append(f"{m.role}: {txt}")
        return "\n\n".join(msgs)


class ConversationRecord:
    conversations: List[Conversation] = []
    user: str = ""
    title: str = ""

    def __init__(
        self,
        *,
        title: str,
        user: str,
        conversations: List[Conversation] = [],
    ):
        self.user = user
        self.title = title
        self.conversations = []
        for c in conversations:
            self.conversations.append(deepcopy(c))

    def to_json(self):
        cr = {
            "conversations": [],
            "user": self.user,
            "title": self.title,
        }
        for conv in self.conversations:
            c = {
                "messages": [dict(m) for m in conv.messages],
                "model_config": dict(conv.model_config),
            }
            if conv.feedback:
                c["feedback"] = dict(conv.feedback)
            cr["conversations"].append(c)
        return json.dumps(cr)

    @classmethod
    def from_json(cls, raw_json: str):
        d = json.loads(raw_json, strict=False)
        cr = ConversationRecord(
            title=d["title"],
            user=d["user"],
        )
        for c in d["conversations"]:
            conversation = Conversation()
            conversation.model_config = ModelConfig.parse_obj(c["model_config"])
            conversation.messages = [
                Message.parse_obj(m) for m in c["messages"]
            ]
            if "feedback" in c:
                conversation.feedback = ConversationFeedback.parse_obj(
                    c["feedback"]
                )
            cr.conversations.append(conversation)
        return cr
