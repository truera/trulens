from trulens.nn.backend import Backend

# Absolute floating tolerance levels.
ATOL_DETERMINISTIC = 1e-08
ATOL_NONDETERMINISTIC = 1e-03
# NOTE(piotrm): if testing using CUDA, this needs to be more tolerant hence the difference.


def tolerance(B) -> float:
    if B.is_deterministic():
        return ATOL_DETERMINISTIC
    else:
        return ATOL_NONDETERMINISTIC
