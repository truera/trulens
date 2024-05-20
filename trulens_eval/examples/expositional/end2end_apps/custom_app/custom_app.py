"""Custom app example.

This app does not make any external network requests or hard work but has some
delays and allocs to mimic the effects of such things.
"""

from concurrent.futures import wait
import random
from typing import Tuple

from examples.expositional.end2end_apps.custom_app.custom_agent import \
    CustomAgent
from examples.expositional.end2end_apps.custom_app.custom_llm import CustomLLM
from examples.expositional.end2end_apps.custom_app.custom_memory import \
    CustomMemory
from examples.expositional.end2end_apps.custom_app.custom_reranker import \
    CustomReranker
from examples.expositional.end2end_apps.custom_app.custom_retriever import \
    CustomRetriever
from examples.expositional.end2end_apps.custom_app.custom_tool import \
    CustomStackTool
from examples.expositional.end2end_apps.custom_app.custom_tool import \
    CustomTool
from examples.expositional.end2end_apps.custom_app.dummy import Dummy

from trulens_eval.tru_custom_app import instrument
from trulens_eval.utils.threading import ThreadPoolExecutor

instrument.method(CustomMemory, "remember")

class CustomTemplate(Dummy):
    """Simple template class that fills in a question and answer."""

    def __init__(self, template, **kwargs):
        super().__init__(**kwargs)

        self.template = template

    @instrument
    def fill(self, question, answer):
        """Fill in the template with the question and answer."""

        return self.template[:] \
            .replace("{question}", question) \
            .replace("{answer}", answer)


class CustomApp(Dummy):
    """Dummy app implementation.
    
    Contains:
    - A memory component.

    - A retriever component.

    - A language model component.

    - A template component.

    - A few tools. Each is a random string->string method that is applied to
      retrieved chunks.

    - A reranker component.

    - A few agents. Be careful about these as these replicate the custom app
      itself and might produce infinite loops.
    """

    def __init__(self, num_agents: int = 2, **kwargs):
        super().__init__(**kwargs)
        
        self.memory = CustomMemory(**kwargs)

        self.retriever = CustomRetriever(**kwargs)

        self.llm = CustomLLM(**kwargs)

        self.template = CustomTemplate(
            "The answer to {question} is probably {answer} or something ..."
        )

        self.tools = [CustomTool(**kwargs) for _ in range(3)] + [CustomStackTool(**kwargs)]

        self.agents = [
            CustomAgent(app=CustomApp(num_agents=0), description=f"ensamble agent {i}") for i in range(num_agents)
        ]

        self.reranker = CustomReranker(**kwargs)

        self.dummy_allocate()

        # Tasks ?

    @instrument
    def process_chunk_by_random_tool(
        self,
        chunk_and_score: Tuple[str, float]
    ) -> str:
        return self\
            .tools[random.randint(0, len(self.tools) - 1)]\
            .invoke(chunk_and_score[0])

    @instrument
    def get_context(self, input: str):
        """Invoke and process contexts retrieval."""

        chunks = self.retriever.retrieve_chunks(input)

        chunks = self.reranker.rerank(
            query_text=input,
            chunks=chunks,
            chunk_scores=None
        ) if self.reranker else chunks

        # Creates a few threads to process chunks in parallel to test apps that
        # make use of threads.
        ex = ThreadPoolExecutor(max_workers=max(1, len(chunks)))

        futures = list(
            ex.submit(self.process_chunk_by_random_tool, chunk_and_score=chunk)
            for chunk in chunks
        )

        wait(futures)
        chunks = list(future.result() for future in futures)

        return chunks

    @instrument
    def respond_to_query(self, query: str):
        """Respond to a query. This is the main method."""

        # Run the agents on the same input. These perform the same steps as this app.
        for agent in self.agents:
            agent.invoke(query)

        # Get rerankined, process chunks.
        chunks = self.get_context(query)

        # Do some remembering.
        self.memory.remember(query)

        # Do some generation.
        answer = self.llm.generate(",".join(chunks))

        # Do some templating.
        output = self.template.fill(question=query, answer=answer)

        # Do some more remembering.
        self.memory.remember(output)

        return output
