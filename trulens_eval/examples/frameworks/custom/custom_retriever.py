from trulens_eval.tru_custom_app import instrument

class CustomRetriever:

    @instrument
    def retrieve_chunks(self, data):
        return [
            f"Relevant chunk: {data.upper()}",
            f"Relevant chunk: {data[::-1]}"
        ]