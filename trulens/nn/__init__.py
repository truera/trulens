_TRULENS_BACKEND = None


def initialize_backend(be):
    global _TRULENS_BACKEND
    _TRULENS_BACKEND = be
    from trulens import backend
