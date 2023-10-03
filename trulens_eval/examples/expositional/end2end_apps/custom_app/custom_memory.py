from time import sleep

from trulens_eval.tru_custom_app import instrument


class CustomMemory:

    def __init__(self):
        self.messages = []

    def remember(self, data):
        self.messages.append(data)
