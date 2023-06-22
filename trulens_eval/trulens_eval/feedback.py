"""
# Feedback Functions
"""

from datetime import datetime
from inspect import Signature
from inspect import signature
import itertools
import logging
from multiprocessing.pool import AsyncResult
import re
from typing import (Any, Callable, Dict, Iterable, Optional, Type, Union)

import numpy as np
import openai
import pydantic
from tqdm.auto import tqdm

from trulens_eval import feedback_prompts
from trulens_eval.keys import *
from trulens_eval.provider_apis import Endpoint
from trulens_eval.schema import Cost
from trulens_eval.schema import FeedbackCall
from trulens_eval.schema import FeedbackDefinition
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import FeedbackResultID
from trulens_eval.schema import FeedbackResultStatus
from trulens_eval.schema import AppDefinition
from trulens_eval.schema import Select
from trulens_eval.db import JSON
from trulens_eval.db import Record
from trulens_eval.provider_apis import HuggingfaceEndpoint, OpenAIEndpoint
from trulens_eval.util import FunctionOrMethod
from trulens_eval.util import jsonify
from trulens_eval.util import OptionalImports
from trulens_eval.util import REQUIREMENT_LANGCHAIN
from trulens_eval.util import SerialModel
from trulens_eval.util import TP, UNICODE_CHECK

with OptionalImports(message=REQUIREMENT_LANGCHAIN):
    from langchain.callbacks import get_openai_callback

PROVIDER_CLASS_NAMES = ['OpenAI', 'Huggingface', 'Cohere']

default_pass_fail_color_threshold = 0.5

logger = logging.getLogger(__name__)


def check_provider(cls_or_name: Union[Type, str]) -> None:
    if isinstance(cls_or_name, str):
        cls_name = cls_or_name
    else:
        cls_name = cls_or_name.__name__

    assert cls_name in PROVIDER_CLASS_NAMES, f"Unsupported provider class {cls_name}"


class Feedback(FeedbackDefinition):
    # Implementation, not serializable, note that FeedbackDefinition contains
    # `implementation` meant to serialize the below.
    imp: Optional[Callable] = pydantic.Field(exclude=True)

    # Aggregator method for feedback functions that produce more than one
    # result.
    agg: Optional[Callable] = pydantic.Field(exclude=True)

    def __init__(
        self,
        imp: Optional[Callable] = None,
        agg: Optional[Callable] = None,
        **kwargs
    ):
        """
        A Feedback function container.

        Parameters:
        
        - imp: Optional[Callable] -- implementation of the feedback function.
        """

        agg = agg or np.mean

        if imp is not None:
            # These are for serialization to/from json and for db storage.
            kwargs['implementation'] = FunctionOrMethod.of_callable(
                imp, loadable=True
            )
            
        else:
            if "implementation" in kwargs:
                imp: Callable = FunctionOrMethod.pick(
                    **(kwargs['implementation'])
                ).load() if kwargs['implementation'] is not None else None

        if agg is not None:
            try:
                # These are for serialization to/from json and for db storage.
                kwargs['aggregator'] = FunctionOrMethod.of_callable(
                    agg, loadable=True
                )
            except:
                # User defined functions in script do not have a module so cannot be serialized
                pass
        else:
            if 'arrgregator' in kwargs:
                agg: Callable = FunctionOrMethod.pick(**(kwargs['aggregator'])
                                                     ).load()

        super().__init__(**kwargs)

        self.imp = imp
        self.agg = agg

        # Verify that `imp` expects the arguments specified in `selectors`:
        if self.imp is not None:
            sig: Signature = signature(self.imp)
            for argname in self.selectors.keys():
                assert argname in sig.parameters, (
                    f"{argname} is not an argument to {self.imp.__name__}. "
                    f"Its arguments are {list(sig.parameters.keys())}."
                )

    def on_input_output(self):
        return self.on_input().on_output()

    def on_default(self):
        ret = Feedback().parse_obj(self)
        ret._default_selectors()
        return ret

    def _print_guessed_selector(self, par_name, par_path):
        if par_path == Select.RecordCalls:
            alias_info = f" or `Select.RecordCalls`"
        elif par_path == Select.RecordInput:
            alias_info = f" or `Select.RecordInput`"
        elif par_path == Select.RecordOutput:
            alias_info = f" or `Select.RecordOutput`"
        else:
            alias_info = ""

        print(
            f"{UNICODE_CHECK} In {self.name}, input {par_name} will be set to {par_path}{alias_info} ."
        )

    def _default_selectors(self):
        """
        Fill in default selectors for any remaining feedback function arguments.
        """

        assert self.imp is not None, "Feedback function implementation is required to determine default argument names."

        sig: Signature = signature(self.imp)
        par_names = list(
            k for k in sig.parameters.keys() if k not in self.selectors
        )

        if len(par_names) == 1:
            # A single argument remaining. Assume it is record output.
            selectors = {par_names[0]: Select.RecordOutput}
            self._print_guessed_selector(par_names[0], Select.RecordOutput)

        elif len(par_names) == 2:
            # Two arguments remaining. Assume they are record input and output
            # respectively.
            selectors = {
                par_names[0]: Select.RecordInput,
                par_names[1]: Select.RecordOutput
            }
            self._print_guessed_selector(par_names[0], Select.RecordInput)
            self._print_guessed_selector(par_names[1], Select.RecordOutput)
        else:
            # Otherwise give up.

            raise RuntimeError(
                f"Cannot determine default paths for feedback function arguments. "
                f"The feedback function has signature {sig}."
            )

        self.selectors = selectors

    @staticmethod
    def evaluate_deferred(tru: 'Tru'):
        db = tru.db

        def prepare_feedback(row):
            record_json = row.record_json
            record = Record(**record_json)

            app_json = row.app_json

            feedback = Feedback(**row.feedback_json)
            feedback.run_and_log(
                record=record,
                app=app_json,
                tru=tru,
                feedback_result_id=row.feedback_result_id
            )

        feedbacks = db.get_feedback()

        for i, row in feedbacks.iterrows():
            if row.status == FeedbackResultStatus.NONE:
                tqdm.write(f"Starting run for row {i}.")

                TP().runlater(prepare_feedback, row)

            elif row.status in [FeedbackResultStatus.RUNNING]:
                now = datetime.now().timestamp()
                if now - row.last_ts > 30:
                    tqdm.write(
                        f"Incomplete row {i} last made progress over 30 seconds ago. Retrying."
                    )
                    TP().runlater(prepare_feedback, row)

                else:
                    tqdm.write(
                        f"Incomplete row {i} last made progress less than 30 seconds ago. Giving it more time."
                    )

            elif row.status in [FeedbackResultStatus.FAILED]:
                now = datetime.now().timestamp()
                if now - row.last_ts > 60 * 5:
                    tqdm.write(
                        f"Failed row {i} last made progress over 5 minutes ago. Retrying."
                    )
                    TP().runlater(prepare_feedback, row)

                else:
                    tqdm.write(
                        f"Failed row {i} last made progress less than 5 minutes ago. Not touching it for now."
                    )

            elif row.status == FeedbackResultStatus.DONE:
                pass

    def __call__(self, *args, **kwargs) -> Any:
        assert self.imp is not None, "Feedback definition needs an implementation to call."
        return self.imp(*args, **kwargs)

    def aggregate(self, func: Callable) -> 'Feedback':
        return Feedback(imp=self.imp, selectors=self.selectors, agg=func)

    @staticmethod
    def of_feedback_definition(f: FeedbackDefinition):
        implementation = f.implementation
        aggregator = f.aggregator

        imp_func = implementation.load()
        agg_func = aggregator.load()

        return Feedback(imp=imp_func, agg=agg_func, **f.dict())

    def _next_unselected_arg_name(self):
        if self.imp is not None:
            sig = signature(self.imp)
            par_names = list(
                k for k in sig.parameters.keys() if k not in self.selectors
            )
            return par_names[0]
        else:
            raise RuntimeError(
                "Cannot determine name of feedback function parameter without its definition."
            )

    def on_prompt(self, arg: Optional[str] = None):
        """
        Create a variant of `self` that will take in the main app input or
        "prompt" as input, sending it as an argument `arg` to implementation.
        """

        new_selectors = self.selectors.copy()

        if arg is None:
            arg = self._next_unselected_arg_name()
            self._print_guessed_selector(arg, Select.RecordInput)

        new_selectors[arg] = Select.RecordInput

        return Feedback(imp=self.imp, selectors=new_selectors, agg=self.agg)

    on_input = on_prompt

    def on_response(self, arg: Optional[str] = None):
        """
        Create a variant of `self` that will take in the main app output or
        "response" as input, sending it as an argument `arg` to implementation.
        """

        new_selectors = self.selectors.copy()

        if arg is None:
            arg = self._next_unselected_arg_name()
            self._print_guessed_selector(arg, Select.RecordOutput)

        new_selectors[arg] = Select.RecordOutput

        return Feedback(imp=self.imp, selectors=new_selectors, agg=self.agg)

    on_output = on_response

    def on(self, *args, **kwargs):
        """
        Create a variant of `self` with the same implementation but the given
        selectors. Those provided positionally get their implementation argument
        name guessed and those provided as kwargs get their name from the kwargs
        key.
        """

        new_selectors = self.selectors.copy()
        new_selectors.update(kwargs)

        for path in args:
            argname = self._next_unselected_arg_name()
            new_selectors[argname] = path
            self._print_guessed_selector(argname, path)

        return Feedback(imp=self.imp, selectors=new_selectors, agg=self.agg)

    def run(
        self, app: Union[AppDefinition, JSON], record: Record
    ) -> FeedbackResult:
        """
        Run the feedback function on the given `record`. The `app` that
        produced the record is also required to determine input/output argument
        names.

        Might not have a AppDefinitionhere but only the serialized app_json .
        """

        if isinstance(app, AppDefinition):
            app_json = jsonify(app)
        else:
            app_json = app

        result_vals = []

        feedback_calls = []

        feedback_result = FeedbackResult(
            feedback_definition_id=self.feedback_definition_id,
            record_id=record.record_id,
            name=self.name
        )

        try:
            cost = Cost()

            for ins in self.extract_selection(app=app_json, record=record):
                
                result_val, part_cost = Endpoint.track_all_costs_tally(lambda: self.imp(**ins))
                cost += part_cost
                result_vals.append(result_val)

                feedback_call = FeedbackCall(args=ins, ret=result_val)
                feedback_calls.append(feedback_call)

            result_vals = np.array(result_vals)
            result = self.agg(result_vals)

            feedback_result.update(
                result=result,
                status=FeedbackResultStatus.DONE,
                cost=cost,
                calls=feedback_calls
            )

            return feedback_result

        except Exception as e:
            raise e

    def run_and_log(
        self,
        record: Record,
        tru: 'Tru',
        app: Union[AppDefinition, JSON] = None,
        feedback_result_id: Optional[FeedbackResultID] = None
    ) -> FeedbackResult:
        record_id = record.record_id
        app_id = record.app_id

        db = tru.db

        # Placeholder result to indicate a run.
        feedback_result = FeedbackResult(
            feedback_definition_id=self.feedback_definition_id,
            feedback_result_id=feedback_result_id,
            record_id=record_id,
            name=self.name
        )

        if feedback_result_id is None:
            feedback_result_id = feedback_result.feedback_result_id

        try:
            db.insert_feedback(
                feedback_result.update(
                    status=FeedbackResultStatus.RUNNING  # in progress
                )
            )

            feedback_result = self.run(
                app=app, record=record
            ).update(feedback_result_id=feedback_result_id)

        except Exception as e:
            db.insert_feedback(
                feedback_result.update(
                    error=str(e), status=FeedbackResultStatus.FAILED
                )
            )
            return

        # Otherwise update based on what Feedback.run produced (could be success or failure).
        db.insert_feedback(feedback_result)

        return feedback_result

    @property
    def name(self):
        """
        Name of the feedback function. Presently derived from the name of the
        function implementing it.
        """

        if self.imp is None:
            raise RuntimeError("This feedback function has no implementation.")

        return self.imp.__name__

    def extract_selection(
        self, app: Union[AppDefinition, JSON], record: Record
    ) -> Iterable[Dict[str, Any]]:
        """
        Given the `app` that produced the given `record`, extract from
        `record` the values that will be sent as arguments to the implementation
        as specified by `self.selectors`.
        """

        arg_vals = {}

        for k, v in self.selectors.items():
            if isinstance(v, Select.Query):
                q = v

            else:
                raise RuntimeError(f"Unhandled selection type {type(v)}.")

            if q.path[0] == Select.Record.path[0]:
                o = record.layout_calls_as_app()
            elif q.path[0] == Select.App.path[0]:
                o = app
            else:
                raise ValueError(
                    f"Query {q} does not indicate whether it is about a record or about a app."
                )

            q_within_o = Select.Query(path=q.path[1:])
            arg_vals[k] = list(q_within_o(o))

        keys = arg_vals.keys()
        vals = arg_vals.values()

        assignments = itertools.product(*vals)

        for assignment in assignments:
            yield {k: v for k, v in zip(keys, assignment)}


pat_1_10 = re.compile(r"\s*([1-9][0-9]*)\s*")


def _re_1_10_rating(str_val):
    matches = pat_1_10.fullmatch(str_val)
    if not matches:
        # Try soft match
        matches = re.search('[1-9][0-9]*', str_val)
        if not matches:
            logger.warn(f"1-10 rating regex failed to match on: '{str_val}'")
            return -10  # so this will be reported as -1 after division by 10

    return int(matches.group())


class Provider(SerialModel):
    class Config:
        arbitrary_types_allowed = True

    endpoint: Optional[Endpoint]


class OpenAI(Provider):
    model_engine: str = "gpt-3.5-turbo"

    # Exclude is important here so that pydantic doesn't try to
    # serialize/deserialize the constant fixed endpoint we need.
    endpoint: Endpoint = pydantic.Field(default_factory=OpenAIEndpoint, exclude=True)

    def __init__(self, **kwargs):
        """
        A set of OpenAI Feedback Functions.

        Parameters:

        - model_engine (str, optional): The specific model version. Defaults to
          "gpt-3.5-turbo".
        """

        super().__init__(**kwargs)  # need to include pydantic.BaseModel.__init__

        set_openai_key()
        
    """
    def to_json(self) -> Dict:
        return Provider.to_json(self, model_engine=self.model_engine)
    """

    def _moderation(self, text: str):
        return self.endpoint.run_me(
            lambda: openai.Moderation.create(input=text)
        )

    def moderation_not_hate(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is hate
        speech.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "hate" and 1 being "not
            hate".
        """
        openai_response = self._moderation(text)
        return 1 - float(
            openai_response["results"][0]["category_scores"]["hate"]
        )

    def moderation_not_hatethreatening(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is
        threatening speech.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "threatening" and 1 being
            "not threatening".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["hate/threatening"]
        )

    def moderation_not_selfharm(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        self harm.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "self harm" and 1 being "not
            self harm".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["self-harm"]
        )

    def moderation_not_sexual(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is sexual
        speech.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "sexual" and 1 being "not
            sexual".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["sexual"]
        )

    def moderation_not_sexualminors(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        sexual minors.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "sexual minors" and 1 being
            "not sexual minors".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["sexual/minors"]
        )

    def moderation_not_violence(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        violence.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "violence" and 1 being "not
            violence".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["violence"]
        )

    def moderation_not_violencegraphic(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        graphic violence.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "graphic violence" and 1
            being "not graphic violence".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["violence/graphic"]
        )

    def qs_relevance(self, question: str, statement: str) -> float:
        """
        Uses OpenAI's Chat Completion App. A function that completes a
        template to check the relevance of the statement to the question.

        Parameters:
            question (str): A question being asked. statement (str): A statement
            to the question.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        return _re_1_10_rating(
            self.endpoint.run_me(
                lambda: openai.ChatCompletion.create(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role":
                                "system",
                            "content":
                                str.format(
                                    feedback_prompts.QS_RELEVANCE,
                                    question=question,
                                    statement=statement
                                )
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def relevance(self, prompt: str, response: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the relevance of the response to a prompt.

        Parameters:
            prompt (str): A text prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        return _re_1_10_rating(
            self.endpoint.run_me(
                lambda: openai.ChatCompletion.create(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role":
                                "system",
                            "content":
                                str.format(
                                    feedback_prompts.PR_RELEVANCE,
                                    prompt=prompt,
                                    response=response
                                )
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def model_agreement(self, prompt: str, response: str) -> float:
        """
        Uses OpenAI's Chat GPT Model. A function that gives Chat GPT the same
        prompt and gets a response, encouraging truthfulness. A second template
        is given to Chat GPT with a prompt that the original response is
        correct, and measures whether previous Chat GPT's response is similar.

        Parameters:
            prompt (str): A text prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not in agreement" and 1
            being "in agreement".
        """
        oai_chat_response = OpenAI().endpoint_openai.run_me(
            lambda: openai.ChatCompletion.create(
                model=self.model_engine,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": feedback_prompts.CORRECT_SYSTEM_PROMPT
                    }, {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )["choices"][0]["message"]["content"]
        )
        agreement_txt = _get_answer_agreement(
            prompt, response, oai_chat_response, self.model_engine
        )
        return _re_1_10_rating(agreement_txt) / 10

    def sentiment(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the sentiment of some text.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "negative sentiment" and 1
            being "positive sentiment".
        """

        return _re_1_10_rating(
            self.endpoint.run_me(
                lambda: openai.ChatCompletion.create(
                    model=self.model_engine,
                    temperature=0.5,
                    messages=[
                        {
                            "role": "system",
                            "content": feedback_prompts.SENTIMENT_SYSTEM_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        )


def _get_answer_agreement(prompt, response, check_response, model_engine):
    print("DEBUG")
    print(feedback_prompts.AGREEMENT_SYSTEM_PROMPT % (prompt, response))
    print("MODEL ANSWER")
    print(check_response)
    oai_chat_response = OpenAI().endpoint.run_me(
        lambda: openai.ChatCompletion.create(
            model=model_engine,
            temperature=0.5,
            messages=[
                {
                    "role":
                        "system",
                    "content":
                        feedback_prompts.AGREEMENT_SYSTEM_PROMPT %
                        (prompt, response)
                }, {
                    "role": "user",
                    "content": check_response
                }
            ]
        )["choices"][0]["message"]["content"]
    )
    return oai_chat_response


# Cannot put these inside Huggingface since it interferes with pydantic.BaseModel.
HUGS_SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
HUGS_TOXIC_API_URL = "https://api-inference.huggingface.co/models/martin-ha/toxic-comment-model"
HUGS_CHAT_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-3B"
HUGS_LANGUAGE_API_URL = "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection"


class Huggingface(Provider):

    # Exclude is important here so that pydantic doesn't try to
    # serialize/deserialize the constant fixed endpoint we need.
    endpoint: Endpoint = pydantic.Field(default_factory=HuggingfaceEndpoint, exclude=True)

    def __init__(self, **kwargs):
        """
        A set of Huggingface Feedback Functions. Utilizes huggingface
        api-inference.
        """

        super().__init__(**kwargs)  # need to include pydantic.BaseModel.__init__

    def language_match(self, text1: str, text2: str) -> float:
        """
        Uses Huggingface's papluca/xlm-roberta-base-language-detection model. A
        function that uses language detection on `text1` and `text2` and
        calculates the probit difference on the language detected on text1. The
        function is: `1.0 - (|probit_language_text1(text1) -
        probit_language_text1(text2))`
        
        Parameters:
        
            text1 (str): Text to evaluate.

            text2 (str): Comparative text to evaluate.

        Returns:

            float: A value between 0 and 1. 0 being "different languages" and 1
            being "same languages".
        """

        def get_scores(text):
            payload = {"inputs": text}
            hf_response = self.endpoint.post(
                url=HUGS_LANGUAGE_API_URL, payload=payload, timeout=30
            )
            return {r['label']: r['score'] for r in hf_response}

        max_length = 500
        scores1: AsyncResult[Dict] = TP().promise(
            get_scores, text=text1[:max_length]
        )
        scores2: AsyncResult[Dict] = TP().promise(
            get_scores, text=text2[:max_length]
        )

        scores1: Dict = scores1.get()
        scores2: Dict = scores2.get()

        langs = list(scores1.keys())
        prob1 = np.array([scores1[k] for k in langs])
        prob2 = np.array([scores2[k] for k in langs])
        diff = prob1 - prob2

        l1 = 1.0 - (np.linalg.norm(diff, ord=1)) / 2.0

        return l1

    def positive_sentiment(self, text: str) -> float:
        """
        Uses Huggingface's cardiffnlp/twitter-roberta-base-sentiment model. A
        function that uses a sentiment classifier on `text`.
        
        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "negative sentiment" and 1
            being "positive sentiment".
        """
        max_length = 500
        truncated_text = text[:max_length]
        payload = {"inputs": truncated_text}

        hf_response = self.endpoint.post(
            url=HUGS_SENTIMENT_API_URL, payload=payload
        )

        for label in hf_response:
            if label['label'] == 'LABEL_2':
                return label['score']

    def not_toxic(self, text: str) -> float:
        """
        Uses Huggingface's martin-ha/toxic-comment-model model. A function that
        uses a toxic comment classifier on `text`.
        
        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "toxic" and 1 being "not
            toxic".
        """
        max_length = 500
        truncated_text = text[:max_length]
        payload = {"inputs": truncated_text}
        hf_response = self.endpoint.post(
            url=HUGS_TOXIC_API_URL, payload=payload
        )

        for label in hf_response:
            if label['label'] == 'toxic':
                return label['score']


# cohere
class Cohere(Provider):
    model_engine: str = "large"

    def __init__(self, model_engine='large'):
        super().__init__()  # need to include pydantic.BaseModel.__init__

        Cohere().endpoint = Endpoint(name="cohere")
        self.model_engine = model_engine

    def sentiment(
        self,
        text,
    ):
        return int(
            Cohere().endpoint.run_me(
                lambda: get_cohere_agent().classify(
                    model=self.model_engine,
                    inputs=[text],
                    examples=feedback_prompts.COHERE_SENTIMENT_EXAMPLES
                )[0].prediction
            )
        )

    def not_disinformation(self, text):
        return int(
            Cohere().endpoint.run_me(
                lambda: get_cohere_agent().classify(
                    model=self.model_engine,
                    inputs=[text],
                    examples=feedback_prompts.COHERE_NOT_DISINFORMATION_EXAMPLES
                )[0].prediction
            )
        )
