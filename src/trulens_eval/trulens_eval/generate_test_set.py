# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.benchmark.generate_test_set`
    instead.
"""

from trulens.core.utils import deprecation as deprecation_utils
from trulens.core.utils import imports as import_utils

deprecation_utils.packages_dep_warn()

with import_utils.OptionalImports(
    messages=import_utils.format_import_errors(
        "trulens.benchmark.generate_test_set",
        purpose="generating test sets",
    )
):
    from trulens.benchmark.generate_Test_set import GenerateTestSet
    from trulens.benchmark.generate_test_set import generate_test_set
