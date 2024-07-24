import asyncio
from concurrent.futures import wait
import time

from trulens.core.app.custom import instrument
from trulens.utils.threading import ThreadPoolExecutor

from examples.expositional.end2end_apps.custom_app.custom_llm import CustomLLM
from examples.expositional.end2end_apps.custom_app.custom_memory import \
    CustomMemory
from examples.expositional.end2end_apps.custom_app.custom_retriever import \
    CustomRetriever

instrument.method(CustomRetriever, 'retrieve_chunks')
instrument.method(CustomMemory, 'remember')


class CustomTemplate:

    def __init__(self, template):
        self.template = template

    @instrument
    def fill(self, question, answer):
        return self.template[:] \
            .replace('{question}', question) \
            .replace('{answer}', answer)


class CustomApp:

    def __init__(self, delay: float = 0.05, alloc: int = 1024 * 1024):
        self.delay = delay  # controls how long to delay certain operations to make it look more realistic
        self.alloc = alloc  # controls how much memory to allocate during some operations
        self.memory = CustomMemory(delay=delay / 20.0, alloc=alloc)
        self.retriever = CustomRetriever(delay=delay / 4.0, alloc=alloc)
        self.llm = CustomLLM(delay=delay, alloc=alloc)
        self.template = CustomTemplate(
            'The answer to {question} is probably {answer} or something ...'
        )

    @instrument
    def retrieve_chunks(self, data):
        return self.retriever.retrieve_chunks(data)

    @instrument
    def respond_to_query(self, input):
        chunks = self.retrieve_chunks(input)

        if self.delay > 0.0:
            time.sleep(self.delay)

        # Creates a few threads to process chunks in parallel to test apps that
        # make use of threads.
        ex = ThreadPoolExecutor(max_workers=max(1, len(chunks)))

        futures = list(
            ex.submit(lambda chunk: chunk + ' processed', chunk=chunk)
            for chunk in chunks
        )

        wait(futures)
        chunks = list(future.result() for future in futures)

        self.memory.remember(input)

        answer = self.llm.generate(','.join(chunks))
        output = self.template.fill(question=input, answer=answer)
        self.memory.remember(output)

        return output

    @instrument
    async def arespond_to_query(self, input):
        # fake async call, must return an async token generator and final result

        res = self.respond_to_query(input)

        async def async_generator():
            for tok in res.split(' '):
                if self.delay > 0.0:
                    await asyncio.sleep(self.delay)

                yield tok + ' '

        gen_task = asyncio.Task(async_generator())

        async def collect_gen():
            ret = ''
            async for tok in gen_task:
                ret += tok
            return ret

        return gen_task, collect_gen
