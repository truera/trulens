"""Custom app example.

This app does not make any external network requests or hard work but has some
delays and allocs to mimic the effects of such things.
"""

import asyncio
from concurrent.futures import wait
from typing import List, Optional, Tuple
import logging

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

logger = logging.getLogger(__name__)

instrument.method(CustomMemory, "remember")


class CustomTemplate(Dummy):
    """Simple template class that fills in a question and answer."""

    def __init__(self, template, **kwargs):
        super().__init__(**kwargs)

        self.template = template

    @instrument
    def fill(self, question: str, answer: str) -> str:
        """Fill in the template with the question and answer.
        
        Args:
            question: The question to fill in.
            
            answer: The answer to fill in.
        """

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

    Args:
        num_agents: Number of agents to create.

        use_threads: Whether to use threads for processing chunks. This is for
            testing/tracing threads. Result might be non-deterministic though.
    """

    DEFAULT_USE_THREADS: bool = False

    def __init__(
        self,
        num_agents: int = 2,
        use_threads: Optional[bool] = None,
        **kwargs
    ):
        if use_threads is None:
            use_threads = CustomApp.DEFAULT_USE_THREADS

        super().__init__(**kwargs)

        self.use_threads = use_threads

        self.memory = CustomMemory(**kwargs)

        self.retriever = CustomRetriever(**kwargs)

        self.llm = CustomLLM(**kwargs)

        self.template = CustomTemplate(
            "The answer to {question} is probably {answer} or something ..."
        )

        # Put some tools into the app and make sure one of them is the one that
        # dumps the stack.
        self.tools = [CustomStackTool(**kwargs)
                     ] + [CustomTool(**kwargs) for _ in range(3)]

        self.agents = [
            CustomAgent(
                app=CustomApp(num_agents=0), description=f"ensamble agent {i}"
            ) for i in range(num_agents)
        ]

        self.reranker = CustomReranker(**kwargs)

        self.dummy_allocate()

        # Tasks ?

    @instrument
    def process_chunk_by_tool(
        self, chunk_and_score: Tuple[str, float], tool_num: int = 0
    ) -> str:
        """Process a (retrieved) chunk by a specified tool.
        
        Args:
            chunk_and_score: The chunk to process including its text and score.

            tool_num: The tool number to use.
        """

        return (
            self.tools[tool_num % len(self.tools)].invoke(chunk_and_score[0])
        )

    @instrument
    async def aprocess_chunk_by_tool(
        self, chunk_and_score: Tuple[str, float], tool_num: int = 0
    ) -> str:
        """Process a (retrieved) chunk by a specified tool.
        
        Args:
            chunk_and_score: The chunk to process including its text and score.

            tool_num: The tool number to use.
        """

        return (
            await
            self.tools[tool_num % len(self.tools)].ainvoke(chunk_and_score[0])
        )

    @instrument
    def get_context(self, query: str) -> List[str]:
        """Invoke and process contexts retrieval.
        
        Args:
            query: The input to retrieve context for.
        """

        chunks = self.retriever.retrieve_chunks(query)

        chunks = self.reranker.rerank(
            query_text=query, chunks=chunks, chunk_scores=None
        ) if self.reranker else chunks

        if self.use_threads:
            # Creates a few threads to process chunks in parallel to test apps
            # that make use of threads.
            ex = ThreadPoolExecutor(max_workers=max(1, len(chunks)))

            futures = list(
                ex.submit(
                    self.process_chunk_by_tool,
                    chunk_and_score=chunk,
                    tool_num=i
                ) for i, chunk in enumerate(chunks)
            )

            wait(futures)
            chunks = list(future.result() for future in futures)
        else:
            # Non-threading but eterministic processing.
            chunks = list(
                self.process_chunk_by_tool(i, chunk)
                for chunk in enumerate(chunks)
            )

        return chunks

    @instrument
    async def aget_context(self, query: str) -> List[str]:
        """Invoke and process contexts retrieval.
        
        Args:
            query: The input to retrieve context for.
        """

        chunks = await self.retriever.aretrieve_chunks(query)

        chunks = await self.reranker.arerank(
            query_text=query, chunks=chunks, chunk_scores=None
        ) if self.reranker else chunks

        if self.use_threads:
            logger.warning(
                "use_threads=True but threads are not used in the async version of get_context."
            )

        # TODO: how to make this deterministic for testing against golden sets?
        tasks = list(
            asyncio.create_task(self.aprocess_chunk_by_tool(chunk_and_score=chunk,tool_num=i))
            for i, chunk in enumerate(chunks)
        )

        results, _ = await asyncio.wait(tasks)

        chunks = list(task.result() for task in results)

        return chunks

    @instrument
    def respond_to_query(self, query: str) -> str:
        """Respond to a query.
        
        This is the main method.
        
        Args:
            query: The query to respond to.
        """

        # Run the agents on the same input. These perform the same steps as this
        # app.
        for agent in self.agents:
            agent.invoke(query)

        # Get rerankined, process chunks.
        chunks = self.get_context(query)

        # Do some remembering.
        self.memory.remember(query)

        # Do some generation.
        answer = self.llm.generate(
            ",".join(filter(lambda c: len(c) < 128, chunks))
        )
        # Skip sthe large chunk coming from CustomStackTool.

        # Do some templating.
        output = self.template.fill(question=query, answer=answer)

        # Do some more remembering.
        self.memory.remember(output)

        return output
    
    @instrument
    async def arespond_to_query(self, query: str) -> str:
        """Respond to a query.
        
        This is the main method.
        
        Args:
            query: The query to respond to.
        """

        # Run the agents on the same input. These perform the same steps as this
        # app.
        for agent in self.agents:
            await agent.ainvoke(query)

        # Get rerankined, process chunks.
        chunks = await self.aget_context(query)

        # Do some remembering.
        await self.memory.aremember(query)

        # Do some generation.
        answer = await self.llm.agenerate(
            ",".join(filter(lambda c: len(c) < 128, chunks))
        )
        # Skip sthe large chunk coming from CustomStackTool.

        # Do some templating.
        output = self.template.fill(question=query, answer=answer)

        # Do some more remembering.
        await self.memory.aremember(output)

        return output
