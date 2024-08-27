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
        "trulens-benchmark",
        purpose="generating test sets",
    )
) as opt:
    from trulens.benchmark.generate import generate_test_set as mod_generate


opt.assert_installed(mod_generate)

GenerateTestSet = mod_generate.GenerateTestSet
