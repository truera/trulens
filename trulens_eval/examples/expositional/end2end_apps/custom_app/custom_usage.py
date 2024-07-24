from trulens.core import TruCustomApp

from examples.frameworks.custom.custom_app import CustomApp

ca = CustomApp()
tru_recorder = TruBasicApp(ca, feedbacks=[], instrument_langchain=False)

with tru_recorder as recording:
    ca.respond_to_query('What is the capital of Indonesia?')

response, record = tru_recorder.with_record(
    ca, 'What is the capital of Indonesia?'
)
