from trulens_eval import TruApp

class CustomRetriever:
    @TruApp.instrument
    def retrieve_chunks(self, data):
        return []