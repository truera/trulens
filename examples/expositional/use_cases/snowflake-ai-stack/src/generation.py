import openai
from langchain_core.documents import Document
from typing import List

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()

class ChatModel:
    def __init__(self, generation_model_name=None):
        # Use the provided model name or default to "gpt4o-mini"
        self.generation_model_name = generation_model_name or "gpt-4o-mini-2024-07-18"
        # Ensure your OpenAI API key is configured in your environment (e.g., OPENAI_API_KEY)

    def construct_prompt(self, question: str, context: list[Document], message_history: List = None) -> List:
        docs_content = "\n\n".join(doc for doc in context)
        system_message = "You are a helpful assistant who answers questions as completely as possible using the context provided. Always respond in paragraph form, and never as a list or enumerated items."
        user_message = f"Question: {question}\n\nContext:\n{docs_content}"
        messages = [
            {
                "role": "system",
                "content": system_message,
            },
        ]
        if message_history:
            messages.extend(message_history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def generate_answer(self, messages: List) -> str:
        response = openai_client.chat.completions.create(
            model=self.generation_model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
        )
        return response.choices[0].message.content

    def generate_stream(self, messages: List):
        response = openai_client.chat.completions.create(
            model=self.generation_model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            yield delta.content
