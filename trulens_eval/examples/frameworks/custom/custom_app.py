from trulens_eval import TruApp
from custom_retriever import CustomRetriever 


class CustomApp:
    retriever: CustomRetriever

    def __init__(self):
        self.retriever = CustomRetriever()

    @TruApp.instrument
    def retrieve_chunks(self, data):
        return self.retriever.retrieve_chunks(data)
    
    @TruApp.instrument
    def respond_to_query(self, input):
        self.retrieve_chunks(input)
        output = input
        return output

