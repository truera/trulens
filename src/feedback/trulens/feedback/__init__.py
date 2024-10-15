# WARNING: This file does not follow the no-init aliases import standard.

from importlib.metadata import version

from trulens.core.utils import imports as import_utils
from trulens.feedback.groundtruth import GroundTruthAggregator
from trulens.feedback.groundtruth import GroundTruthAgreement
from trulens.feedback.llm_provider import LLMProvider

with import_utils.OptionalImports(
    messages=import_utils.format_import_errors(
        ["llama-index", "scikit-learn"], "using embedding feedback functions"
    )
) as opt:
    from trulens.feedback.embeddings import Embeddings

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)


__all__ = [
    "GroundTruthAgreement",
    "GroundTruthAggregator",
    "LLMProvider",
    "Embeddings",
]
