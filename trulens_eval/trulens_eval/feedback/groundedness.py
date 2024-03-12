import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.base import Provider
from trulens_eval.feedback.provider.hugs import Huggingface
from trulens_eval.utils.generated import re_0_10_rating
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_BEDROCK
from trulens_eval.utils.imports import REQUIREMENT_GROUNDEDNESS
from trulens_eval.utils.imports import REQUIREMENT_LITELLM
from trulens_eval.utils.imports import REQUIREMENT_OPENAI
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.serial import SerialModel

with OptionalImports(messages=REQUIREMENT_GROUNDEDNESS):
    from nltk.tokenize import sent_tokenize

with OptionalImports(messages=REQUIREMENT_BEDROCK):
    from trulens_eval.feedback.provider.bedrock import Bedrock

with OptionalImports(messages=REQUIREMENT_OPENAI):
    from trulens_eval.feedback.provider.openai import AzureOpenAI
    from trulens_eval.feedback.provider.openai import OpenAI

with OptionalImports(messages=REQUIREMENT_LITELLM):
    from trulens_eval.feedback.provider.litellm import LiteLLM

logger = logging.getLogger(__name__)


class Groundedness(WithClassInfo, SerialModel):
    """
    Measures Groundedness.

    Currently the groundedness
    functions work well with a summarizer. This class will use an LLM to
    find the relevant strings in a text. The groundedness_provider can
    either be an LLM provider (such as OpenAI) or NLI with huggingface.

    Usage:
        ```python
        from trulens_eval.feedback import Groundedness
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = OpenAI()
        groundedness_imp = Groundedness(groundedness_provider=openai_provider)
        ```

    Usage:
        ```python
        from trulens_eval.feedback import Groundedness
        from trulens_eval.feedback.provider.hugs import Huggingface
        huggingface_provider = Huggingface()
        groundedness_imp = Groundedness(groundedness_provider=huggingface_provider)
        ```

    Args:
        groundedness_provider: Provider to use for evaluating groundedness. This
            should be [OpenAI][trulens_eval.feedback.provider.openai.OpenAI] LLM
            or [HuggingFace][trulens_eval.feedback.provider.hugs.Huggingface]
            NLI. Defaults to `OpenAI`.
    """

    groundedness_provider: Provider

    def __init__(
        self, groundedness_provider: Optional[Provider] = None, **kwargs
    ):
        if groundedness_provider is None:
            logger.warning("Provider not provided. Using OpenAI.")
            groundedness_provider = OpenAI()

        super().__init__(groundedness_provider=groundedness_provider, **kwargs)

    def groundedness_measure_with_cot_reasons(
        self, source: str, statement: str
    ) -> Tuple[float, dict]:
        """A measure to track if the source material supports each sentence in
        the statement using an LLM provider.

        The LLM will process the entire statement at once, using chain of
        thought methodology to emit the reasons. 

        Usage on RAG Contexts:
            ```python
            from trulens_eval import Feedback
            from trulens_eval.feedback import Groundedness
            from trulens_eval.feedback.provider.openai import OpenAI
            grounded = feedback.Groundedness(groundedness_provider=OpenAI())

            f_groundedness = feedback.Feedback(grounded.groundedness_measure_with_cot_reasons).on(
                Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content # See note below
            ).on_output().aggregate(grounded.grounded_statements_aggregator)
            ```

            The `on(...)` selector can be changed. See [Feedback Function Guide : Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)

        Args:
            source: The source that should support the statement.

            statement: The statement to check groundedness.

        Returns:
            A measure between 0 and 1, where 1 means each sentence is grounded in the source.
        """
        groundedness_scores = {}
        if not isinstance(self.groundedness_provider, LLMProvider):
            raise AssertionError(
                "Only LLM providers are supported for groundedness_measure_with_cot_reasons."
            )
        else:
            reason = self.groundedness_provider._groundedness_doc_in_out(
                source, statement
            )
            i = 0
            for line in reason.split('\n'):
                if "Score" in line:
                    groundedness_scores[f"statement_{i}"
                                       ] = re_0_10_rating(line) / 10
                    i += 1
        return groundedness_scores, {"reasons": reason}

    def groundedness_measure_with_nli(self, source: str,
                                      statement: str) -> Tuple[float, dict]:
        """
        A measure to track if the source material supports each sentence in the statement using an NLI model.

        First the response will be split into statements using a sentence tokenizer.The NLI model will process each statement using a natural language inference model, and will use the entire source.

        Usage on RAG Contexts:
        ```
        from trulens_eval import Feedback
        from trulens_eval.feedback import Groundedness
        from trulens_eval.feedback.provider.hugs = Huggingface
        grounded = feedback.Groundedness(groundedness_provider=Huggingface())


        f_groundedness = feedback.Feedback(grounded.groundedness_measure_with_nli).on(
            Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content # See note below
        ).on_output().aggregate(grounded.grounded_statements_aggregator)
        ```
        The `on(...)` selector can be changed. See [Feedback Function Guide : Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)


        Args:
            source (str): The source that should support the statement
            statement (str): The statement to check groundedness

        Returns:
            float: A measure between 0 and 1, where 1 means each sentence is grounded in the source.
            str: 
        """
        groundedness_scores = {}
        if not isinstance(self.groundedness_provider, Huggingface):
            raise AssertionError(
                "Only Huggingface provider is supported for groundedness_measure_with_nli"
            )
        else:
            reason = ""
            if isinstance(source, list):
                source = ' '.join(map(str, source))
            hypotheses = sent_tokenize(statement)
            for i, hypothesis in enumerate(tqdm(
                    hypotheses, desc="Groundendess per statement in source")):
                score = self.groundedness_provider._doc_groundedness(
                    premise=source, hypothesis=hypothesis
                )
                reason = reason + str.format(
                    prompts.GROUNDEDNESS_REASON_TEMPLATE,
                    statement_sentence=hypothesis,
                    supporting_evidence="[Doc NLI Used full source]",
                    score=score * 10,
                )
                groundedness_scores[f"statement_{i}"] = score
        return groundedness_scores, {"reason": reason}

    def groundedness_measure(self, source: str,
                             statement: str) -> Tuple[float, dict]:
        """
        Groundedness measure is deprecated in place of the chain-of-thought version. Defaulting to groundedness_measure_with_cot_reasons.
        """
        logger.warning(
            "groundedness_measure is deprecated, please use groundedness_measure_with_cot_reasons or groundedness_measure_with_nli instead."
        )
        return self.groundedness_measure_with_cot_reasons(source, statement)

    def groundedness_measure_with_summarize_step(
        self, source: str, statement: str
    ) -> float:
        """A measure to track if the source material supports each sentence in the statement. 
        This groundedness measure is more accurate; but slower using a two step process.
        - First find supporting evidence with an LLM
        - Then for each statement sentence, check groundendness
        
        Usage on RAG Contexts:
        ```
        from trulens_eval import Feedback
        from trulens_eval.feedback import Groundedness
        from trulens_eval.feedback.provider.openai import OpenAI
        grounded = feedback.Groundedness(groundedness_provider=OpenAI())


        f_groundedness = feedback.Feedback(grounded.groundedness_measure_with_summarize_step).on(
            Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content # See note below
        ).on_output().aggregate(grounded.grounded_statements_aggregator)
        ```
        The `on(...)` selector can be changed. See [Feedback Function Guide : Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)


        Args:
            source (str): The source that should support the statement
            statement (str): The statement to check groundedness

        Returns:
            float: A measure between 0 and 1, where 1 means each sentence is grounded in the source.
        """
        groundedness_scores = {}
        if not isinstance(self.groundedness_provider, LLMProvider):
            raise AssertionError(
                "Only LLM providers are supported for groundedness_measure_with_cot_reasons."
            )
        else:
            reason = ""
            hypotheses = sent_tokenize(statement)
            for i, hypothesis in enumerate(tqdm(
                    hypotheses, desc="Groundendess per statement in source")):
                score = self.groundedness_provider._groundedness_doc_in_out(
                    premise=source, hypothesis=hypothesis
                )
                supporting_premise = self.groundedness_provider._find_relevant_string(
                    source, hypothesis
                )
                score = self.groundedness_provider._summarized_groundedness(
                    premise=supporting_premise, hypothesis=hypothesis
                )
                reason = reason + str.format(
                    prompts.GROUNDEDNESS_REASON_TEMPLATE,
                    statement_sentence=hypothesis,
                    supporting_evidence=supporting_premise,
                    score=score * 10,
                )
                groundedness_scores[f"statement_{i}"] = score
        return groundedness_scores, {"reason": reason}

    def grounded_statements_aggregator(
        self, source_statements_multi_output: List[Dict]
    ) -> float:
        """Aggregates multi-input, mulit-output information from the groundedness_measure methods.

        Args:
            source_statements_multi_output (List[Dict]): A list of scores. Each list index is a context. The Dict is a per statement score.

        Returns:
            float: for each statement, gets the max groundedness, then averages over that.
        """
        all_results = []

        statements_to_scores = {}

        # Ensure source_statements_multi_output is a list
        if not isinstance(source_statements_multi_output, list):
            source_statements_multi_output = [source_statements_multi_output]

        for multi_output in source_statements_multi_output:
            for k in multi_output:
                if k not in statements_to_scores:
                    statements_to_scores[k] = []
                statements_to_scores[k].append(multi_output[k])

        for k in statements_to_scores:
            all_results.append(np.max(statements_to_scores[k]))

        return np.mean(all_results)
