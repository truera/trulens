import logging
from typing import Dict, List

import numpy as np
from tqdm.auto import tqdm

from trulens_eval.feedback import prompts
from trulens_eval.feedback.provider import Provider
from trulens_eval.feedback.provider.hugs import Huggingface
from trulens_eval.feedback.provider.openai import AzureOpenAI
from trulens_eval.feedback.provider.openai import OpenAI
from trulens_eval.utils.serial import SerialModel
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.generated import re_1_10_rating

logger = logging.getLogger(__name__)


class Groundedness(SerialModel, WithClassInfo):
    summarize_provider: Provider
    groundedness_provider: Provider

    def __init__(
        self,
        summarize_provider: Provider = None,
        groundedness_provider: Provider = None
    ):
        """Instantiates the groundedness providers. Currently the groundedness functions work well with a summarizer.
        This class will use an OpenAI summarizer to find the relevant strings in a text. The groundedness_provider can 
        either be an llm with OpenAI or NLI with huggingface.

        Args:
            groundedness_provider (Provider, optional): groundedness provider options: OpenAI LLM or HuggingFace NLI. Defaults to OpenAI().
        """
        if summarize_provider is None:
            summarize_provider = OpenAI()
        if groundedness_provider is None:
            groundedness_provider = OpenAI()
        if not isinstance(groundedness_provider,
                          (OpenAI, AzureOpenAI, Huggingface)):
            raise Exception(
                "Groundedness is only supported groundedness_provider as OpenAI, AzureOpenAI or Huggingface Providers."
            )
        super().__init__(
            summarize_provider=summarize_provider,
            groundedness_provider=groundedness_provider,
            obj=self  # for WithClassInfo
        )

    def groundedness_measure(self, source: str, statement: str) -> float:
        """A measure to track if the source material supports each sentence in the statement. 
        This groundedness measure is faster; but less accurate than `groundedness_measure_with_summarize_step` 

        ```
        grounded = feedback.Groundedness(groundedness_provider=OpenAI())


        f_groundedness = feedback.Feedback(grounded.groundedness_measure).on(
            Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content
        ).on_output().aggregate(grounded.grounded_statements_aggregator)
        ```
        Args:
            source (str): The source that should support the statement
            statement (str): The statement to check groundedness

        Returns:
            float: A measure between 0 and 1, where 1 means each sentence is grounded in the source.
        """
        groundedness_scores = {}
        if isinstance(self.groundedness_provider, (AzureOpenAI, OpenAI)):
            plausible_junk_char_min = 4  # very likely "sentences" under 4 characters are punctuation, spaces, etc
            if len(statement) > plausible_junk_char_min:
                reason = self.summarize_provider._groundedness_doc_in_out(
                    source, statement
                )
            i = 0
            for line in reason.split('\n'):
                if "Score" in line:
                    groundedness_scores[f"statement_{i}"
                                       ] = re_1_10_rating(line) / 10
                    i += 1
            return groundedness_scores, {"reason": reason}
        if isinstance(self.groundedness_provider, Huggingface):
            reason = ""
            for i, hypothesis in enumerate(
                    tqdm(statement.split("."),
                         desc="Groundendess per statement in source")):
                plausible_junk_char_min = 4  # very likely "sentences" under 4 characters are punctuation, spaces, etc
                if len(hypothesis) > plausible_junk_char_min:
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

    def groundedness_measure_with_summarize_step(
        self, source: str, statement: str
    ) -> float:
        """A measure to track if the source material supports each sentence in the statement. 
        This groundedness measure is more accurate; but slower using a two step process.
        - First find supporting evidence with an LLM
        - Then for each statement sentence, check groundendness
        ```
        grounded = feedback.Groundedness(groundedness_provider=OpenAI())


        f_groundedness = feedback.Feedback(grounded.groundedness_measure_with_summarize_step).on(
            Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content
        ).on_output().aggregate(grounded.grounded_statements_aggregator)
        ```
        Args:
            source (str): The source that should support the statement
            statement (str): The statement to check groundedness

        Returns:
            float: A measure between 0 and 1, where 1 means each sentence is grounded in the source.
        """
        groundedness_scores = {}
        reason = ""
        for i, hypothesis in enumerate(
                tqdm(statement.split("."),
                     desc="Groundendess per statement in source")):
            plausible_junk_char_min = 4  # very likely "sentences" under 4 characters are punctuation, spaces, etc
            if len(hypothesis) > plausible_junk_char_min:
                supporting_premise = self.summarize_provider._find_relevant_string(
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
            source_statements_multi_output (np.ndarray): a 2D array with the first dimension corresponding to a source text,
                and the second dimension corresponding to each sentence in a statement; it's groundedness score

        Returns:
            float: for each statement, gets the max groundedness, then averages over that.
        """
        all_results = []
        for multi_output in source_statements_multi_output:
            result_vals = list(multi_output.values())
        all_results.append(result_vals)
        all_results = np.asarray(all_results)
        max_over_sources = np.max(all_results, axis=0)
        return np.mean(max_over_sources)
