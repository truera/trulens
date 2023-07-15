from trulens_eval import Provider

# Globally importable classes/functions to be used for testing feedback
# functions.


def custom_feedback_function(t1: str) -> float:
    return 0.1


class CustomProvider(Provider):

    @staticmethod
    def custom_provider_static_method(t1: str) -> float:
        return 0.2

    def custom_provider_method(self, t1: str) -> float:
        return 0.3


class CustomClass():

    @staticmethod
    def custom_class_static_method(t1: str) -> float:
        return 0.4

    def custom_class_method(self, t1: str) -> float:
        return 0.5