import re
from typing import Any, AsyncIterator, List, Optional

from feedback import f_small_local_models_context_relevance
from feedback import get_provider
import replicate
from retrieve import AVAILABLE_RETRIEVERS
from schema import Conversation
from schema import Message
from schema import ModelConfig
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from trulens_eval.guardrails.base import context_filter
# replicate key for running model
from trulens_eval.tru_custom_app import instrument

PROVIDER_MODELS = {
    "Replicate":
        {
            "Snowflake Arctic Instruct": "snowflake/snowflake-arctic-instruct",
            "LLaMa 3 8B": "meta/meta-llama-3-8b",
            "Mistral 7B Instruct (v0.2)": "mistralai/mistral-7b-instruct-v0.2",
        },
    "Cortex":
        {
            "Snowflake Arctic": "snowflake-arctic",
            "LLaMa 3 8B": "llama3-8b",
            "Mistral 7B": "mistral-7b",
        }
}


def encode_arctic(messages: List[Message]):
    prompt = []
    for msg in messages:
        prompt.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")

    prompt.append("<|im_start|>assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    return prompt_str


def encode_llama3(messages: List[Message]):
    prompt = []
    prompt.append("<|begin_of_text|>")
    for msg in messages:
        prompt.append(f"<|start_header_id|>{msg.role}<|end_header_id|>")
        prompt.append(f"{msg.content.strip()}<|eot_id|>")
    prompt.append("<|start_header_id|>assistant<|end_header_id|>")
    prompt.append("")
    prompt_str = "\n\n".join(prompt)
    return prompt_str


def encode_generic(messages: List[Message]):
    prompt = []
    for msg in messages:
        prompt.append(f"{msg.role}:\n" + msg.content)

    prompt.append("assistant:")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    return prompt_str


def _reencode_outputs(s: str):
    return s.encode('utf-8', 'surrogateescape').decode('ISO-8859-1')


ENCODING_MAPPING = {
    "Snowflake Arctic": encode_arctic,
    "LLaMa 3 8B": encode_llama3,
    "Mistral 7B": encode_generic,
}


class StreamGenerator:

    def get_last_user_message(self, prompt_str):
        # Regex to find the last 'user' message
        match = re.findall(
            r'<\|im_start\|>user\n(.*?)(?=<\|im_end\|>)', prompt_str, re.DOTALL
        )
        if match:
            return match[-1].strip()
        return ""

    def prepare_prompt(self, conversation: Conversation):
        messages = conversation.messages
        model_config = conversation.model_config

        if model_config.system_prompt:
            system_msg = Message(
                role="system", content=model_config.system_prompt
            )
            messages = [system_msg] + messages

        prompt_str = ENCODING_MAPPING[model_config.model](messages)

        # Extract the last user message from the prompt string
        message = None
        for message in messages[::-1]:
            if message.role == "user":
                break
        last_user_message = message.content if message and message.role == "user" else ""
        return last_user_message, prompt_str

    def _generate_stream_with_replicate(
        self, model_name: str, model_input: dict
    ):
        stream_iter: AsyncIterator = replicate.stream(
            model_name, input=model_input
        )
        for t in stream_iter:
            yield str(t)

    def _generate_with_cortex(self, model_name: str, model_input: dict):
        response = get_provider("Cortex")._create_chat_completion(
            prompt=model_input['prompt'],
            temperature=model_input['temperature'],
            model=model_name
        )
        yield str(response)

    def _write_stream_to_st(
        self,
        stream_iter: AsyncIterator,
        st_container: Optional[DeltaGenerator] = None
    ) -> str:
        full_text_response = ""

        if st_container is None:
            full_text_response = st.write_stream(stream_iter)
        else:
            full_text_response = st_container.write_stream(stream_iter)

        return full_text_response

    def _generate_response(
        self, model_config: ModelConfig, model_input: dict[str, Any]
    ):
        full_model_name = PROVIDER_MODELS[model_config.provider][
            model_config.model]

        if model_config.provider == "Replicate":
            stream_iter = self._generate_stream_with_replicate(
                full_model_name, model_input
            )
        elif model_config.provider == "Cortex":
            stream_iter = self._generate_with_cortex(
                full_model_name, model_input
            )
        else:
            raise ValueError(f"Invalid Provider {model_config.provider}")
        return stream_iter

    @instrument
    def generate_response(
        self,
        last_user_message: str,
        prompt_str: str,
        conversation: Conversation,
        st_container: Optional[DeltaGenerator] = None,
        should_write=True
    ) -> str:
        model_config = conversation.model_config

        final_response = ""
        model_input = {
            "prompt": prompt_str,
            "prompt_template": r"{prompt}",
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
        }

        stream_iter = self._generate_response(
            model_config=model_config, model_input=model_input
        )

        if should_write:
            final_response = self._write_stream_to_st(stream_iter, st_container)
        else:
            for chunk in stream_iter:
                final_response += chunk

        return _reencode_outputs(final_response)

    @instrument
    def retrieve_context(
        self, last_user_message: str, prompt_str: str,
        conversation: Conversation
    ):

        @context_filter(
            f_small_local_models_context_relevance,
            conversation.model_config.retrieval_filter
        )
        def retrieve():
            retriever = AVAILABLE_RETRIEVERS[conversation.model_config.retriever
                                            ]
            texts = retriever.retrieve(query=last_user_message)
            context_message = "\n\n".join(texts)
            return _reencode_outputs(context_message), texts

        return retrieve()

    @instrument
    def retrieve_and_generate_response(
        self,
        last_user_message: str,
        prompt_str: str,
        conversation: Conversation,
        st_container: Optional[DeltaGenerator] = None
    ) -> str:
        context_message, nodes = self.retrieve_context(
            last_user_message=last_user_message,
            prompt_str=prompt_str,
            conversation=conversation
        )  # Fixed by passing the conversation object instead of prompt_str
        model_config = conversation.model_config

        full_prompt = (
            "We have provided context information below. \n"
            "---------------------\n"
            f"{context_message}"
            "\n---------------------\n"
            f"Given this information, please answer the question: {prompt_str}"
            "\n---------------------\n"
            "Only respond if the answer is supported by the information above. Otherwise, respond that you don't have the required information available."
        )

        model_input = {
            "prompt": full_prompt,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
        }

        final_response = ""
        stream_iter = self._generate_response(
            model_config=model_config, model_input=model_input
        )
        final_response = self._write_stream_to_st(stream_iter, st_container)
        return _reencode_outputs(final_response)
