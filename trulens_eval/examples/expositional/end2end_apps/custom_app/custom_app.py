import asyncio
from asyncio import sleep
from concurrent.futures import wait

from examples.expositional.end2end_apps.custom_app.custom_llm import CustomLLM
from examples.expositional.end2end_apps.custom_app.custom_memory import \
    CustomMemory
from examples.expositional.end2end_apps.custom_app.custom_retriever import \
    CustomRetriever

from trulens_eval.tru_custom_app import instrument
from trulens_eval.utils.threading import ThreadPoolExecutor

instrument.method(CustomRetriever, "retrieve_chunks")
instrument.method(CustomMemory, "remember")


class CustomTemplate:

    def __init__(self, template):
        self.template = template

    @instrument
    def fill(self, question, answer):
        return self.template[:] \
            .replace("{question}", question) \
            .replace("{answer}", answer)


class CustomApp:

    def __init__(self):
        self.memory = CustomMemory()
        self.retriever = CustomRetriever()
        self.llm = CustomLLM()
        self.template = CustomTemplate(
            "The answer to {question} is probably {answer} or something ..."
        )

    @instrument
    def retrieve_chunks(self, data):
        return self.retriever.retrieve_chunks(data)

    @instrument
    def respond_to_query(self, input):
        chunks = self.retrieve_chunks(input)

        # Creates a few threads to process chunks in parallel to test apps that
        # make use of threads.
        ex = ThreadPoolExecutor(max_workers=max(1, len(chunks)))

        futures = list(
            ex.submit(lambda chunk: chunk + " processed", chunk=chunk) for chunk in chunks
        )

        wait(futures)
        chunks = list(future.result() for future in futures)

        self.memory.remember(input)

        answer = self.llm.generate(",".join(chunks))
        output = self.template.fill(question=input, answer=answer)
        self.memory.remember(output)

        return output

    @instrument
    async def arespond_to_query(self, input):
        # fake async call, must return an async token generator and final result

        res = self.respond_to_query(input)

        async def async_generator():
            for tok in res.split(" "):
                await sleep(0.05)
                yield tok + " "

        gen_task = asyncio.Task(async_generator())

        async def collect_gen():
            ret = ""
            async for tok in gen_task:
                ret += tok
            return ret

        return gen_task, collect_gen
