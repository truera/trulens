from examples.dev.dummy_app.app import DummyApp

from trulens_eval import TruApp

ca = DummyApp()
tru_recorder = TruApp(ca)

with tru_recorder as recording:
    ca.respond_to_query("What is the capital of Indonesia?")

record = recording.get()