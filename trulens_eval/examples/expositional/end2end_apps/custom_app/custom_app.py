import asyncio
from concurrent.futures import wait
import random
from typing import Tuple

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

instrument.method(CustomRetriever, "retrieve_chunks")
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
    - A few tools.
    - A reranker component.
    - A few agents (TODO).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.memory = CustomMemory(**kwargs)

        self.retriever = CustomRetriever(**kwargs)

        self.llm = CustomLLM(**kwargs)

        self.template = CustomTemplate(
            "The answer to {question} is probably {answer} or something ..."
        )

        self.tools = [CustomTool(**kwargs) for _ in range(3)] + [CustomStackTool(**kwargs)]

        self.agents = [] # TODO

        self.reranker = CustomReranker(**kwargs)

        self.dummy_allocate()

        # Tasks ?

    @instrument
    def process_chunk_by_random_tool(self, chunk_and_score: Tuple[str, float]) -> str:
        return self.tools[random.randint(0, len(self.tools) - 1)].invoke(chunk_and_score[0])

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
    def respond_to_query(self, input: str):
        """Respond to a query. This is the main method."""

        # Get rerankined, process chunks.
        chunks = self.get_context(input)

        # Do some remembering.
        self.memory.remember(input)

        # Do some generation.
        answer = self.llm.generate(",".join(chunks))

        # Do some templating.
        output = self.template.fill(question=input, answer=answer)

        # Do some more remembering.
        self.memory.remember(output)

        return output

    @instrument
    async def arespond_to_query(self, input: str):
        """Fake async call, must return an async token generator and final
        result.
        
        TODO: This probably does not work.
        """

        res = self.respond_to_query(input)

        async def async_generator():
            for tok in res.split(" "):
                if self.delay > 0.0:
                    await asyncio.sleep(self.delay)

                yield tok + " "

        gen_task = asyncio.Task(async_generator())

        async def collect_gen():
            ret = ""
            async for tok in gen_task:
                ret += tok
            return ret

        return gen_task, collect_gen
