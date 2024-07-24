from trulens.core.feedback.base_feedback import SkipEval
from trulens.core.feedback.base_provider import Provider

# Globally importable classes/functions to be used for testing feedback
# functions.


def skip_if_odd(val: float):
    """Feedback function that returns its argument as long as it is even and
    raises SkipEval if it is odd.

    This is used to test the SkipEval functionality.
    """

    if int(val) % 2 == 1:
        raise SkipEval(reason=f"Because input value {val} was odd.")

    return float(val)


def custom_feedback_function(t1: str) -> float:
    return 0.1


class CustomProvider(Provider):
    # Provider inherits WithClassInfo and pydantic.BaseModel which means we can
    # deserialize it.

    attr: float

    @staticmethod
    def static_method(t1: str) -> float:
        return 0.2

    @classmethod
    def class_method(cls, t1: str) -> float:
        return 0.3

    def method(self, t1: str) -> float:
        return 0.4 + self.attr


class CustomClassNoArgs:
    # This one is ok as it has no init arguments so we can deserialize it just
    # from its module and name.

    @staticmethod
    def static_method(t1: str) -> float:
        return 0.5

    @classmethod
    def class_method(cls, t1: str) -> float:
        return 0.6

    def method(self, t1: str) -> float:
        return 0.7


class CustomClassWithArgs:
    # These should fail as we don't know how to initialize this class during
    # deserialization.

    def __init__(self, attr: float):
        self.attr = attr

    @staticmethod
    def static_method(t1: str) -> float:
        return 0.8

    @classmethod
    def class_method(cls, t1: str) -> float:
        return 0.9

    def method(self, t1: str) -> float:
        return 1.0 + self.attr


def make_nonglobal_feedbacks():
    # Creates the same stuff as above except in the local scope of this
    # function, making the results not globally importable.

    # TODO: bug here that if the methods below are named the same as the
    # globally importable ones above, they will get imported as them
    # incorrectly.

    class NG:  # "non-global"
        @staticmethod
        def NGcustom_feedback_function(t1: str) -> float:
            return 0.1

        class NGCustomProvider(Provider):
            # Provider inherits WithClassInfo and pydantic.BaseModel which means we can
            # deserialize it.

            attr: float

            @staticmethod
            def static_method(t1: str) -> float:
                return 0.2

            @classmethod
            def class_method(cls, t1: str) -> float:
                return 0.3

            def method(self, t1: str) -> float:
                return 0.4 + self.attr

        class NGCustomClassNoArgs:
            # This one is ok as it has no init arguments so we can deserialize it just
            # from its module and name.

            @staticmethod
            def static_method(t1: str) -> float:
                return 0.5

            @classmethod
            def class_method(cls, t1: str) -> float:
                return 0.6

            def method(self, t1: str) -> float:
                return 0.7

        class NGCustomClassWithArgs:
            # These should fail as we don't know how to initialize this class during
            # deserialization.

            def __init__(self, attr: float):
                self.attr = attr

            @staticmethod
            def static_method(t1: str) -> float:
                return 0.8

            @classmethod
            def class_method(cls, t1: str) -> float:
                return 0.9

            def method(self, t1: str) -> float:
                return 1.0 + self.attr

    return NG
