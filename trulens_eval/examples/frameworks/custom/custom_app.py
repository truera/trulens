from examples.frameworks.custom.custom_llm import CustomLLM
from examples.frameworks.custom.custom_retriever import CustomRetriever

from trulens_eval.tru_custom_app import instrument

instrument.method(CustomRetriever, "retrieve_chunks")


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
        answer = self.llm.generate(",".join(chunks))
        output = self.template.fill(question=input, answer=answer)

        return output
