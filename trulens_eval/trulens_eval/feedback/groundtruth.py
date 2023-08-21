import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pydantic

from trulens_eval.feedback.provider import Provider
from trulens_eval.feedback.provider.openai import OpenAI
from trulens_eval.utils.pyschema import FunctionOrMethod
from trulens_eval.utils.serial import SerialModel
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.generated import re_1_10_rating

logger = logging.getLogger(__name__)


class GroundTruthAgreement(SerialModel, WithClassInfo):
    ground_truth: Union[List[str], FunctionOrMethod]
    provider: Provider

    ground_truth_imp: Optional[Callable] = pydantic.Field(exclude=True)

    def __init__(
        self,
        ground_truth: Union[List[str], Callable, FunctionOrMethod],
        provider: Provider = None
    ):
        if provider is None:
            provider = OpenAI()
        if isinstance(ground_truth, List):
            ground_truth_imp = None
        elif isinstance(ground_truth, FunctionOrMethod):
            ground_truth_imp = ground_truth.load()
        elif isinstance(ground_truth, Callable):
            ground_truth_imp = ground_truth
            ground_truth = FunctionOrMethod.of_callable(ground_truth)
        elif isinstance(ground_truth, Dict):
            # Serialized FunctionOrMethod?
            ground_truth = FunctionOrMethod.pick(**ground_truth)
            ground_truth_imp = ground_truth.load()
        else:
            raise RuntimeError(
                f"Unhandled ground_truth type: {type(ground_truth)}."
            )

        super().__init__(
            ground_truth=ground_truth,
            ground_truth_imp=ground_truth_imp,
            provider=provider,
            obj=self  # for WithClassInfo
        )

    def _find_response(self, prompt: str) -> Optional[str]:
        if self.ground_truth_imp is not None:
            return self.ground_truth_imp(prompt)

        responses = [
            qr["response"] for qr in self.ground_truth if qr["query"] == prompt
        ]
        if responses:
            return responses[0]
        else:
            return None

    def agreement_measure(
        self, prompt: str, response: str
    ) -> Union[float, Tuple[float, Dict[str, str]]]:
        """
        Uses OpenAI's Chat GPT Model. A function that that measures
        similarity to ground truth. A second template is given to Chat GPT
        with a prompt that the original response is correct, and measures
        whether previous Chat GPT's response is similar.

        Parameters:
            prompt (str): A text prompt to an agent. response (str): The
            agent's response to the prompt.

        Returns:
            - float: A value between 0 and 1. 0 being "not in agreement" and 1
                being "in agreement".
            - dict: with key 'ground_truth_response'
        """
        ground_truth_response = self._find_response(prompt)
        if ground_truth_response:
            agreement_txt = self.provider._get_answer_agreement(
                prompt, response, ground_truth_response
            )
            ret = re_1_10_rating(agreement_txt) / 10, dict(
                ground_truth_response=ground_truth_response
            )
        else:
            ret = np.nan

        return ret
