import asyncio
from collections import defaultdict
from concurrent.futures import wait
import logging
from typing import (
    Any, AsyncIterable, Dict, Iterable, List, Optional, Tuple, Type
)

from examples.dev.dummy_app.agent import DummyAgent
from examples.dev.dummy_app.dummy import Dummy
from examples.dev.dummy_app.llm import DummyLLM
from examples.dev.dummy_app.memory import DummyMemory
from examples.dev.dummy_app.reranker import DummyReranker
from examples.dev.dummy_app.retriever import DummyRetriever
from examples.dev.dummy_app.template import DummyTemplate
from examples.dev.dummy_app.tool import DummyStackTool
from examples.dev.dummy_app.tool import DummyTool

from trulens_eval.tru_custom_app import instrument
from trulens_eval.utils.threading import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class DummyApp(Dummy):
    """Dummy app implementation.
    
    Contains:
        - A memory component of type
          [DummyMemory][examples.dev.dummy_app.memory.DummyMemory].

        - A retriever component of type
          [DummyRetriever][examples.dev.dummy_app.retriever.DummyRetriever]

        - A language model component of type
          [DummyLLM][examples.dev.dummy_app.llm.DummyLLM].

        - A template component of type
          [DummyTemplate][examples.dev.dummy_app.template.DummyTemplate].

        - A few tools of type
          [DummyTool][examples.dev.dummy_app.tool.DummyTool]. The first tool is
          one that records the call stack when it gets invokved
          ([DummyStackTool][examples.dev.dummy_app.tool.DummyStackTool]). Each
          of the following ones (is random string->string method that is applied
          to retrieved chunks.

        - A reranker component of type
          [DummyReranker][examples.dev.dummy_app.reranker.DummyReranker]

        - A few agents of type
          [DummyAgent][examples.dev.dummy_app.agent.DummyAgent]. Be careful
          about these as these replicate the custom app itself and might produce
          infinite loops.

    Args:
        num_agents: Number of agents to create.

        num_tools: Number of tools to create.

        use_parallel: Whether to use parallelism for processing retrieval
            chunks. This is either threads or async parallelism depending on
            which method is invoked.

        comp_kwargs: A dictionary of arguments to pass to each component
            constructor. The key is the component class and the value is a
            dictionary of arguments to pass to the constructor.

        **kwargs: Additional arguments passed in to all constructors meant for
            [Dummy][examples.dev.dummy_app.dummy.Dummy] arguments.
    """

    def __init__(
        self,
        num_agents: int = 2,
        num_tools: int = 3,
        use_parallel: bool = False,
        comp_kwargs: Optional[Dict[Type, Dict[str, Any]]] = None,
        **kwargs: Dict[str, Any]
    ):

        super().__init__(**kwargs)

        comp_kwargs = defaultdict(dict, comp_kwargs if comp_kwargs else {})

        self.use_parallel = use_parallel

        self.memory = DummyMemory(**kwargs, **comp_kwargs[DummyMemory])

        self.retriever = DummyRetriever(**kwargs, **comp_kwargs[DummyRetriever])

        self.llm = DummyLLM(**kwargs, **comp_kwargs[DummyLLM])

        self.template = DummyTemplate(
            """Please answer the following question given the context.
QUESTION: {question}
CONTEXT: {context}
""", **kwargs, **comp_kwargs[DummyTemplate]
        )

        # Put some tools into the app and make sure the first is the one that
        # dumps the stack.
        self.tools = [DummyStackTool(**kwargs, **comp_kwargs[DummyStackTool])
                     ] + [
                         DummyTool(**kwargs, **comp_kwargs[DummyTool])
                         for _ in range(num_tools - 1)
                     ]

        self.agents = [
            DummyAgent(
                app=DummyApp(num_agents=0),
                description=f"ensamble agent {i}",
                **kwargs,
                **comp_kwargs[DummyAgent]
            ) for i in range(num_agents)
        ]

        self.reranker = DummyReranker(**kwargs, **comp_kwargs[DummyReranker])

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

        if self.use_parallel:
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
            # Non-parallel but deterministic processing.
            chunks = list(
                self.process_chunk_by_tool(tool_num=i, chunk_and_score=chunk)
                for i, chunk in enumerate(chunks)
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

        if self.use_parallel:
            # TODO: how to make this deterministic for testing against golden
            # sets?
            tasks = list(
                asyncio.create_task(
                    self.
                    aprocess_chunk_by_tool(chunk_and_score=chunk, tool_num=i)
                ) for i, chunk in enumerate(chunks)
            )

            results, _ = await asyncio.wait(tasks)

            chunks = list(task.result() for task in results)
        else:
            # Non-parallel but deterministic processing.
            chunks = list(
                self.process_chunk_by_tool(tool_num=i, chunk_and_score=chunk)
                for i, chunk in enumerate(chunks)
            )

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
        summary = self.llm.generate(
            ",".join(filter(lambda c: len(c) < 128, chunks))
        )

        # Do some templating.
        answer_prompt = self.template.fill(question=query, context=summary)

        # Another llm call to "get the final answer".
        answer = self.llm.generate(answer_prompt)

        # Do some more remembering.
        self.memory.remember(answer)

        return answer

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
        summary = await self.llm.agenerate(
            ",".join(filter(lambda c: len(c) < 128, chunks))
        )

        # Do some templating.
        answer_prompt = self.template.fill(question=query, context=summary)

        # Another llm call to "get the final answer".
        answer = await self.llm.agenerate(answer_prompt)

        # Do some more remembering.
        self.memory.remember(answer)

        return answer

    @instrument
    def stream_respond_to_query(self, query: str) -> Iterable[str]:
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
        summary = self.llm.generate(
            ",".join(filter(lambda c: len(c) < 128, chunks))
        )

        # Do some templating.
        answer_prompt = self.template.fill(question=query, context=summary)

        # Another llm call to "get the final answer". Streaming this time.
        answer = ""
        for chunk in self.llm.stream(answer_prompt):
            answer += chunk
            yield chunk

        # Do some more remembering.
        self.memory.remember(answer)

    @instrument
    async def astream_respond_to_query(self, query: str) -> AsyncIterable[str]:
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
        summary = await self.llm.agenerate(
            ",".join(filter(lambda c: len(c) < 128, chunks))
        )

        # Do some templating.
        answer_prompt = self.template.fill(question=query, context=summary)

        # Another llm call to "get the final answer". Streaming this time.
        answer = ""
        async for chunk in self.llm.astream(answer_prompt):
            answer += chunk
            yield chunk

        # Do some more remembering.
        self.memory.remember(answer)
