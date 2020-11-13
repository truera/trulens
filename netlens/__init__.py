_NETLENS_BACKEND = None


def initialize_backend(be):
    global _NETLENS_BACKEND
    _NETLENS_BACKEND = be
    from netlens import backend
