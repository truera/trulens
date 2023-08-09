from trulens_eval.tru_custom_app import instrument
from custom_retriever import CustomRetriever 


class CustomApp:

    def __init__(self):
        self.retriever = CustomRetriever()

    @instrument
    def retrieve_chunks(self, data):
        return self.retriever.retrieve_chunks(data)

    @instrument
    def respond_to_query(self, input):
        chunks = self.retrieve_chunks(input)
        output = f"The answer to {input} is probably {chunks[0]} or something ..."
        return output
