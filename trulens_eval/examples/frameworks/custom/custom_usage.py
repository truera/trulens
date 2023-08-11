from custom_module import CustomApp

from trulens_eval import TruApp

ca = CustomApp()
ta = TruApp(ca, feedbacks=[], instrument_langchain=False)

ta.respond_to_query("What is the capital of Indonesia?")
response, record = ta.respond_to_query_with_record(
    "What is the capital of Indonesia?"
)
