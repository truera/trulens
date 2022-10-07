from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import tempfile
from typing import (Any, Callable, Iterable, List, Optional, Set, Tuple,
                    TypeVar, Union)

import numpy as np

from trulens.nn.attribution import AttributionMethod
from trulens.nn.backend import get_backend
from trulens.nn.models._model_base import \
    ModelWrapper  # todo: move this class to somewhere else
from trulens.nn.quantities import ClassQoI
from trulens.nn.quantities import ComparativeQoI
from trulens.nn.quantities import MaxClassQoI
from trulens.nn.quantities import QoI
from trulens.utils import tru_logger
from trulens.utils import try_import
from trulens.utils.typing import Inputs
from trulens.utils.typing import ModelInputs
from trulens.utils.typing import nested_cast
from trulens.utils.typing import Outputs
from trulens.utils.typing import Tensor
from trulens.utils.typing import TensorLike
from trulens.utils.typing import Tensors
from trulens.utils.typing import Uniform
from trulens.vis import arrays_different
from trulens.vis import COLORMAP
from trulens.vis import ColorMap


class NLPOutput(ABC):
    """Base class for visualization output formats."""

    # Element type
    E = TypeVar("E")

    # Rendered output type
    R = TypeVar("R")

    @abstractmethod
    def blank(self) -> E:
        ...

    @abstractmethod
    def space(self) -> E:
        ...

    @abstractmethod
    def big(self, s: E) -> E:
        ...

    @abstractmethod
    def highlight(self, s: E) -> E:
        ...

    @abstractmethod
    def sub(self, e: E, color: str) -> E:
        ...

    def scores(
        self, scores: np.ndarray, labels: List[str], qoi: QoI = None
    ) -> E:
        if sum(scores) != 1.0:
            scores = np.exp(scores) / np.exp(scores).sum()

        highlights = [False] * len(scores)

        color = "rgba(0, 100, 100, 1.0)"

        content = []

        pred = np.argmax(scores)

        if isinstance(qoi, ClassQoI):
            highlights[qoi.cl] = color
            content += [
                self.label(f"ClassQoI", color=color),
                self.sub(str(qoi.cl), color=color),
                self.space()
            ]

        if isinstance(qoi, MaxClassQoI):
            highlights = [color] * len(scores)
            content += [self.label(f"MaxClassQoI", color=color), self.space()]

        if isinstance(qoi, ComparativeQoI):
            highlights[qoi.cl1] = "green"
            highlights[qoi.cl2] = "red"
            content += [
                self.label(f"ComparativeQoI", color=color),
                self.sub(str(qoi.cl1), color="green"),
                self.space(),
                self.sub(str(qoi.cl2), color="red"),
                self.space()
            ]

        for i, (score, label, highlight) in enumerate(zip(scores, labels,
                                                          highlights)):
            temp = self.magnitude_colored(label, mag=score)

            if highlight is not None:
                temp = self.highlight(temp, color=highlight)

            if pred == i:
                temp = self.highlight(temp, color="white")

            content += [temp]

            if i + 1 < len(scores):
                content += [self.space()]

        #content += [self.label(f")")]

        return self.concat(*content)

    @abstractmethod
    def token(self, s: str, token_id=None) -> E:
        ...

    @abstractmethod
    def label(self, s: str) -> E:
        ...

    @abstractmethod
    def line(self, e: E) -> E:
        ...

    @abstractmethod
    def magnitude_colored(self, s: str, mag: float, color_map: COLORMAP) -> E:
        ...

    @abstractmethod
    def concat(self, *parts: Iterable[E]) -> E:
        ...

    @abstractmethod
    def render(self, e: E) -> R:
        ...

    @abstractmethod
    def open(self, r: R) -> None:
        ...


# TODO(piotrm): implement a latex output format


class EnvType:
    ...


class Term(EnvType):
    ...


class Jupyter(EnvType):
    ...


class Colab(Jupyter):
    ...


def guess_env_type():
    # From Andreas
    '''
    Tests whether current process is running in a:
            o terminal as a regular Python shell
            o jupyter notebook
            o Google colab
    returns one of {'terminal', 'jupyter', 'colab', None}
    None means could not determine.
    '''
    try:
        from IPython import get_ipython
    except ImportError:
        return Term()
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return Jupyter()  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return Term()  # Terminal running IPython
        elif shell == 'Shell' and get_ipython(
        ).__class__.__module__ == 'google.colab._shell':
            return Colab()
        else:
            return Term()  # Other type (?)
    except NameError:
        return Term()  # Probably standard Python interpreter


class PlainText(NLPOutput):
    """Plain text visualization output format."""

    E = str
    R = str

    def blank(self) -> E:
        return ""

    def space(self) -> E:
        return " "

    def big(self, s: E) -> E:
        return f"_{s}_"

    def highlight(self, s: E, color: str) -> E:
        return f"[{s}]"

    def sub(self, s: E, color: str) -> E:
        return f".{s}."

    def token(self, s: str, token_id=None) -> E:
        s = self.label(s)

        content = s

        if token_id is not None:
            content += f"({token_id})"

        return content

    def label(self, s: str, color: str) -> E:
        return s

    def line(self, e: E) -> E:
        return e

    def magnitude_colored(
        self, s: str, mag: float, color_map: COLORMAP = ColorMap.default
    ) -> E:
        return f"{self.label(s)}({mag:0.3f})"

    def concat(self, *parts: Iterable[E]) -> E:
        return ''.join(parts)

    def render(self, e: E) -> R:
        return e

    def open(self, r: R) -> None:
        raise NotImplementedError


class HTML(NLPOutput):
    """HTML visualization output format."""

    E = 'domonic.dom.Node'
    R = str

    def __init__(self):
        self.m_html_util = try_import("html", msg="html output")
        self.m_dom = try_import("domonic.dom", msg="html output")
        self.m_html = try_import("domonic", msg="html output")

    def blank(self) -> E:
        return self.m_dom.Document.createDocumentFragment()

    def space(self) -> E:
        return self.m_dom.Text("&nbsp;")

    def label(self, s: str, color: str = "white") -> E:
        return self.m_html.span(
            self.m_dom.Text(self.m_html_util.escape(s)),
            style=f"color: {color};" if color is not None else ""
        )

    def quote(self, e: E, mark:str='"', background: str = "black", color: str = "white") -> E:
        if isinstance(e, str):
            e = self.label(self.m_html_util.escape(e), color=color)

        return self.m_html.div(
            [
                self.m_html.div(mark, style=f"text-align: center; line-height: 0.5; width: 4%; font-size: 3.0em; padding: 5px; color: gray; display: inline-block; border: 0px solid white; vertical-align: top; "),
                self.m_html.div(e, style=f"display: inline-block; max-width: 88%;  border: 0px solid white; color: {color};"),
                self.m_html.div(mark, style=f"text-align: center; line-height: 0.5; width: 4%; font-size: 3.0em; padding: 5px; color: gray; display: inline-block; border: 0px solid white; vertical-align: bottom; "),
            ],
            style=f"color: {color}; border: 0px solid white; padding: 5px; background: {background};"
        )

    def linebreak(self) -> E:
        return self.m_html.br()

    def line(self, e: E) -> E:
        return self.m_html.div(
            e, style="padding: 5px; maring: 0px; background: black; overflow-wrap: break-word; line-height: 3.0;"
        )

    def big(self, e: E) -> E:
        return self.m_html.strong(e)

    def highlight(self, s: E, color: str = "white") -> E:
        return self.m_html.span(
            s, style=f"padding: 2px; border: 2px solid {color};"
        )

    def sub(self, e: E, color: str = "white") -> E:
        return self.m_html.sub(e, style=f"color: {color};")

    def token(self, s: str, token_id=None) -> E:
        s = self.label(s, color=None)

        extra_arg = {}
        if token_id is not None:
            extra_arg['title'] = f"token id: {token_id}"

        pad_top = 0
        pad_bot = 2

        return self.m_html.span(
            s,
            style=f'border-top: {pad_top}px solid gray; '
            f'border-bottom: {pad_bot}px solid gray; '
            f'margin-left 1px; '
            f'margin-right: 1px; '
            # f'background: black; '
            f'color: white;',
            **extra_arg
        )

    def scale(self, mag: float, color_map=ColorMap.default) -> E:

        if isinstance(mag, tuple):
            maglabel = f"{mag[1]:0.1f}{mag[0]:0.1f}"
        else:
            maglabel = f"{mag:0.3f}"

        if isinstance(mag, tuple):
            magn = mag[0]
            magp = mag[1]
            rn, gn, bn, an = np.array(color_map(mag[0])) * 255
            rp, gp, bp, ap = np.array(color_map(mag[1])) * 255
        else:
            if mag > 0:
                magp = mag
                magn = 0.0
                rp, gp, bp, ap = np.array(color_map(mag)) * 255
                rn, gn, bn, an = 0, 0, 0, 0
            else:
                magp = 0.0
                magn = mag
                rn, gn, bn, an = np.array(color_map(mag)) * 255
                rp, gp, bp, ap = 0, 0, 0, 0

        s = self.label(maglabel)

        rgban = f"rgba({rn}, {gn}, {bn}, {an})"
        rgbap = f"rgba({rp}, {gp}, {bp}, {ap})"

        pad_top = 0
        pad_bot = 0

        pad_top = int(min(magp, 1.0) * 10)
        pad_bot = int(-max(magn, -1.0) * 10)

        pieces = []

        pieces += [
            self.m_html.span(
                "",
                title=f"{maglabel}",
                style=f'border-right: {pad_bot}px solid {rgban}; '
                f'margin-left: {25-pad_bot}px;'
            )
        ]

        pieces += [
            self.m_html.span(
                s,
                title=f"{maglabel}",
                style=f'display: inline-block; '
                f'width: 60px; '
                f'padding: 2px; '
                f'text-align: center;',
            )
        ]

        pieces += [
            self.m_html.span(
                "",
                title=f"{maglabel}",
                style=f'border-left: {pad_top}px solid {rgbap}; '
                f'margin-right: {25-pad_top}px;'
            )
        ]

        return self.concat(*pieces)

    MagnitudeLike = Union[float, Tuple[float, float]]

    def magnitude_colored(
        self,
        s: str,
        mag: MagnitudeLike,
        color_map=ColorMap.default,
        color='white'
    ) -> E:
        if isinstance(mag, tuple):
            maglabel = f"{mag[1]:0.2f}{mag[0]:0.2f}"
            magn = mag[0]
            magp = mag[1]
            rn, gn, bn, an = np.array(color_map(mag[0])) * 255
            rp, gp, bp, ap = np.array(color_map(mag[1])) * 255
        else:
            maglabel = f"{mag:0.3f}"
            if mag > 0:
                magp = mag
                magn = 0.0
                rp, gp, bp, ap = np.array(color_map(mag)) * 255
                rn, gn, bn, an = 0, 0, 0, 0
            else:
                magp = 0.0
                magn = mag
                rn, gn, bn, an = np.array(color_map(mag)) * 255
                rp, gp, bp, ap = 0, 0, 0, 0

        s = self.label(s)

        rgban = f"rgba({rn}, {gn}, {bn}, {an})"
        rgbap = f"rgba({rp}, {gp}, {bp}, {ap})"

        pad_top = 0
        pad_bot = 0

        pad_top = int(min(magp, 1.0) * 10)
        pad_bot = int(-max(magn, -1.0) * 10)

        return self.m_html.span(
            s,
            title=f"{maglabel}",
            style=f'border-top: {pad_top}px solid {rgbap}; '
            f'border-bottom: {pad_bot}px solid {rgban}; '
            f'margin-left 1px; '
            f'margin-right: 1px; '
            # f'background: black; '
            f'color={color};'
        )

    def concat(self, *pieces: Iterable[E]) -> E:
        temp = self.blank()
        for piece in pieces:
            temp.appendChild(piece)

        return temp

    def render(self, e: E) -> R:
        return str(self.m_html.html(self.m_html.body(e)))

    def open(self, r):
        mod = try_import("webbrowser", msg="html open")

        # from Andreas

        with tempfile.NamedTemporaryFile(prefix='attrs_', mode='w') as fd:
            fd.write(r)
            mod.open_new_tab(f"file://{fd.name}")


class IPython(HTML):
    """Interactive python visualization output format."""

    def __init__(self):
        super(IPython, self).__init__()

        self.m_ipy = try_import("IPython", "Jupyter output")

    def render(self, e):
        html = HTML.render(self, e)
        return self.m_ipy.display.HTML(html)


@dataclass
class InstanceOptionals:
    # Minimally available when rendering tokens or related.
    texts: str
    input_ids: TensorLike

    # Next two available if model wrapper is known.
    # outputs: TensorLike
    logits: TensorLike

    # Available if attributor is known.
    attributions: Outputs[TensorLike]
    multipliers: TensorLike

    # Available if attributor is known and show_doi is set.
    gradients: Outputs[Uniform[TensorLike]]
    interventions: Uniform[TensorLike]

    # Available if embedder is known.
    embeddings: TensorLike


@dataclass
class InstancesOptionals:
    # Minimally available when rendering tokens or related.
    texts: Inputs[str]
    input_ids: Inputs[TensorLike]

    # Next two available if model wrapper is known.
    # outputs: Inputs[TensorLike] # one for each Input
    logits: Inputs[TensorLike] = None  # one for each Input

    # Available if attributor is known.
    attributions: Inputs[TensorLike] = None
    multipliers: Inputs[TensorLike] = None

    # Available if attributor is known and show_doi is set.
    gradients: Inputs[Uniform[TensorLike]] = None
    interventions: Inputs[Uniform[TensorLike]] = None

    # Available if embedder is known.
    embeddings: Inputs[TensorLike] = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        ret = ""

        for k in dir(self):
            v = getattr(self, k)

            if k not in ["texts", "input_ids", "logits", "attributions",
                         "multipliers", "gradients", "interventions",
                         "embeddings"]:
                continue

            ret += k + "\t: "
            if hasattr(v, "shape"):
                ret += str(v.shape)
            elif hasattr(v, "__len__"):
                ret += f"[{len(v)}]"
            else:
                ret += str(v)

            ret += "\n"

        return ret

    def for_text_index(self, i):
        return InstanceOptionals(
            texts=self.texts[i],
            input_ids=self.input_ids[i],
            # outputs = self.outputs[i],
            logits=self.logits[i] if self.logits is not None else None,
            attributions=self.attributions[i]
            if self.attributions is not None else None,
            multipliers=self.multipliers[i]
            if self.multipliers is not None else None,
            gradients=self.gradients[:,
                                     i] if self.gradients is not None else None,
            interventions=self.interventions[:, i]
            if self.interventions is not None else None,
            embeddings=self.embeddings[i]
            if self.embeddings is not None else None
        )


class NLP(object):
    """NLP Visualization tools."""

    # Batches of text inputs not yet tokenized.
    TextBatch = TypeVar("TextBatch")

    # Inputs that are directly accepted by wrapped models, tokenized.
    # TODO(piotrm): Reuse other typevars/aliases from elsewhere.
    ModelInput = TypeVar("ModelInput")

    # Outputs produced by wrapped models.
    # TODO(piotrm): Reuse other typevars/aliases from elsewhere.
    ModelOutput = TypeVar("ModelOutput")

    def __init__(
        self,
        wrapper: ModelWrapper = None,
        output: Optional[NLPOutput] = None,
        labels: Optional[Iterable[str]] = None,
        tokenize: Optional[Callable[[TextBatch], ModelInputs]] = None,
        embedder: Optional[Any] = None,  # fix type hint
        embeddings: Optional[np.ndarray] = None,
        embedding_distance: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        decode: Optional[Callable[[Tensor], str]] = None,
        input_accessor: Optional[Callable[[ModelInputs],
                                          Iterable[Tensor]]] = None,
        output_accessor: Optional[Callable[[ModelOutput],
                                           Iterable[Tensor]]] = None,
        attr_aggregate: Optional[Callable[[Tensor], Tensor]] = None,
        aggregate_twosided: bool = False,
        hidden_tokens: Optional[Set[int]] = set(),
        color_map: Callable[[float], Tuple[float, float, float,
                                           float]] = ColorMap.default
    ):
        """Initializate NLP visualization tools for a given environment.

        Parameters:
            wrapper: ModelWrapper
                The wrapped model whose channel we're visualizing.

            output: NLPOutput, optional
                Visualization output format. Defaults to PlainText unless
                ipython is detected and in which case defaults to IPython
                format.

            labels: Iterable[str], optional
                Names of prediction classes for classification models.

            tokenize: Callable[[TextBatch], ModelInput], optional
                Method to tokenize an instance.

            decode: Callable[[Tensor], str], optional
                Method to invert/decode the tokenization.

            input_accessor: Callable[[ModelInputs], Iterable[Tensor]], optional
                Method to extract input/token ids from model inputs (tokenize
                output) if needed.

            output_accessor: Callable[[ModelOutput], Iterable[Tensor]], optional
                Method to extract outout logits from output structures if
                needed.

            attr_aggregate: Callable[[Tensor], Tensor], optional
                Method to aggregate attribution for embedding into a single
                value. Defaults to sum.

            hidden_tokens: Set[int], optional
                For token-based visualizations, which tokens to hide.

            color_map: ColorMap
                Means of coloring floats in [-1.0, 1.0]. 

            embeddings: np.ndarray
        """
        if output is None:
            term_type = guess_env_type()
            if isinstance(term_type, Jupyter):
                output = IPython()

            else:
                output = PlainText()
                tru_logger(
                    "WARNING: could not guess preferred visualization output format, using PlainText"
                )

        # TODO: automatic inference of various parameters for common repositories like huggingface, tfhub.

        self.output = output
        self.labels = labels

        self.tokenize = tokenize
        self.decode = decode
        self.wrapper = wrapper

        self.embeddings = embeddings
        if self.embeddings is not None and not isinstance(self.embeddings,
                                                          np.ndarray):
            raise ValueError(
                f"embeddings is expected to be a numpy arrow but {self.embeddings.__class__} was given"
            )

        self.embedder = embedder
        self.embedding_distance = embedding_distance

        if not isinstance(self.embedding_distance, Callable):
            L = lambda ord: lambda emb: np.linalg.norm(
                emb - self.embeddings, ord=ord, axis=1
            )

            if isinstance(self.embedding_distance, str):
                if self.embedding_distance == "l2":
                    self.embedding_distance = L(2.0)
                elif self.embedding_distance == "l1":
                    self.embedding_distance = L(1.0)
                elif self.embedding_distance == "cosine":
                    self.embedding_distance = lambda emb: -np.dot(
                        emb, self.embeddings
                    )
                else:
                    raise ValueError(
                        f"Unknown embedding distance indicator string {self.embedding_distance}"
                    )

            elif self.embedding_distance is None:
                self.embedding_distance = L(2.0)

            else:
                raise ValueError(
                    f"Do not know how to interpret embedding distance = {self.embedding_distance}"
                )

        self.input_accessor = input_accessor  # could be inferred
        self.output_accessor = output_accessor  # could be inferred

        B = get_backend()

        self.aggregate_twosided = aggregate_twosided

        if attr_aggregate is None:
            if self.aggregate_twosided:
                attr_aggregate = lambda t: (B.sum(t[t < 0]), B.sum(t[t > 0]))
            else:
                attr_aggregate = B.sum

        self.attr_aggregate = attr_aggregate

        self.hidden_tokens = hidden_tokens

        self.color_map = color_map

    def token_attribution_scale(self):
        """
        Render an attribution scale.
        """

        cells = [self.output.label("scale:"), self.output.space()]

        for f in range(-10, 11):
            cells.append(
                self.output.magnitude_colored(
                    str(f / 10.0) if f <= 0 else "+" + str(f / 10.0), f / 10.0,
                    self.color_map
                )
            )

        return self.output.line(self.output.concat(*cells))

    def _tokens_stability_instance(
        self,
        opt: InstanceOptionals,
        index: int,
        show_id=False,
        show_doi=False,
        show_text=False,
        highlights=None,
        attributor=None
    ):
        """
        Produce the entire output for a single input instance. This might
        include multiple interpolations if show_doi is enabled.
        """

        B = get_backend()

        sent = [
            self.output.label(f"Instance ", color="gray"),
            self.output.label(str(index)),
            self.output.label(" : ", color="gray"), 
            self.output.space()
        ]

        if self.wrapper is not None:            
            logits = B.as_array(opt.logits)
            # TODO: show pred
            # pred = logits.argmax()

            if len(logits.shape) <= 1:
                
                if self.labels is None:
                    self.labels = list(map(str, range(0, len(logits))))

                sent += [
                    self.output.label("Output/QoI: ", color="gray"),
                    self.output.scores(
                        logits,
                        self.labels,
                        qoi=attributor.qoi if attributor else None
                    )
                ]

                sent += [self.output.linebreak()]

        if opt.attributions is None:
            attr = [None] * len(opt.input_ids)
        else:
            attr = opt.attributions

        effective_input_ids = [opt.input_ids]
        effective_attr = [attr]

        if opt.embeddings is not None:
            effective_embeddings = [opt.embeddings]
        else:
            effective_embeddings = [None] * len(opt.input_ids)

        if show_doi:
            if opt.interventions is None:
                raise ValueError(
                    "show_doi requires return_doi to be enabled on attributor"
                )
            if opt.gradients is None:
                raise ValueError(
                    "show_doi requires return_grads to be enabled on attributor"
                )

            effective_input_ids += [opt.input_ids] * len(opt.interventions)
            effective_embeddings += list(opt.interventions)
            effective_attr += list(opt.gradients)

        if len(effective_attr) > 1:
            #sent += [self.output.linebreak(), self.output.linebreak()]
            pass

        for iid, (sentence_input_ids, sentence_embeddings,
                  sentence_attr) in enumerate(zip(effective_input_ids,
                                                  effective_embeddings,
                                                  effective_attr)):

            interv = []

            if show_text and iid == 0:
                sent += [
                    self.output.quote(opt.texts, background="white", color="black"),
                    #self.output.space(),
                    #self.output.label("=>", color="gray"),
                    #self.output.space()
                ]

            if opt.multipliers is None:
                sentence_multiplier = [None] * len(sentence_input_ids)
            else:
                sentence_multiplier = opt.multipliers

            if sentence_embeddings is None:
                sentence_embeddings = [None] * len(sentence_input_ids)

            if iid > 0:
                interv += [
                    self.output.label("+", color="gray"),
                    self.output.space()
                ]

                mag_sentence = self.attr_aggregate(
                    sentence_attr * sentence_multiplier
                )

                interv += [self.output.scale(mag_sentence), self.output.space()]

            for i, (input_id, emb, attr,
                    mult) in enumerate(zip(sentence_input_ids,
                                           sentence_embeddings, sentence_attr,
                                           sentence_multiplier)):

                if show_doi:
                    input_id, dist = self._closest_token(emb)
                else:
                    if input_id in self.hidden_tokens:
                        continue
                    dist = None

                if self.decode is not None:
                    word = self.decode(input_id)
                else:
                    word = str(input_id)

                if word[0] == ' ':
                    word = word[1:]
                    interv += [self.output.space()]

                if word == "":
                    word = "�"

                cap = lambda x: x

                if highlights is not None and highlights[i]:
                    cap = self.output.big

                if attr is not None:
                    if show_doi and iid > 0:  # as in multiply_activations
                        attr = attr * mult  # TODO: need to consider baseline here as well

                    mag = self.attr_aggregate(attr)

                    interv += [
                        cap(
                            self.output.magnitude_colored(
                                word, mag, color_map=self.color_map
                            )
                        )
                    ]
                else:
                    interv += [cap(self.output.token(word, token_id=input_id))]

                if show_id:
                    interv += [
                        self.output.sub(self.output.label(str(input_id)))
                    ]

            if iid == 0 and len(effective_attr) > 1:
                interv += [
                    self.output.space(),
                    self.output.label(
                        f"= (1/{len(opt.interventions)}) * ", color="gray"
                    )
                ]

            sent += [self.output.quote(self.output.concat(*interv), mark="/", background='teal')]
            #all += [self.output.linebreak()]#, self.output.linebreak()]

        return self.output.concat(self.output.line(self.output.concat(*sent)))

    def _get_optionals(
        self, texts: List[str], attributor: AttributionMethod = None, extra_model_inputs: dict={}
    ):
        B = get_backend()

        def to_numpy(thing):
            return np.array(nested_cast(B, thing, np.ndarray))

        given_inputs = self.tokenize(texts)

        if isinstance(given_inputs, Tensors):
            inputs = given_inputs.as_model_inputs()
        else:
            inputs = ModelInputs(kwargs=given_inputs)

        for k, v in extra_model_inputs.items():
            inputs.kwargs[k] = v

        input_ids = given_inputs
        if self.input_accessor is not None:
            input_ids = self.input_accessor(input_ids)

            if (not isinstance(input_ids, Iterable)) or isinstance(input_ids,
                                                                   dict):
                raise ValueError(
                    f"Inputs ({input_ids.__class__.__name__}) need to be iterable over instances. You might need to set input_accessor."
                )

        # Start constructing the collection of optional fields to return.
        opt = InstancesOptionals(texts=texts, input_ids=to_numpy(input_ids))

        if self.embedder is not None:
            embeddings = self.embedder(input_ids)
            opt.embeddings = to_numpy(embeddings)

        if self.wrapper is not None:
            outputs = inputs.call_on(self.wrapper._model)

            if self.output_accessor is not None:
                opt.logits = self.output_accessor(outputs)
            else:
                opt.logits = outputs

            if (not isinstance(opt.logits, Iterable)) or isinstance(opt.logits,
                                                                    dict):
                raise ValueError(
                    f"Outputs ({opt.logits.__class__.__name__}) need to be iterable over instances. You might need to set output_accessor."
                )

            opt.logits = to_numpy(opt.logits)

        if attributor is not None:
            pieces = attributor._attributions(inputs)

            attributions = pieces.attributions
            if len(attributions) != 1 or len(attributions[0]) != 1:
                raise ValueError(
                    "Only attrubutions with one attribution layer and one qoi output are supported for visualization."
                )

            opt.attributions = to_numpy(attributions)[0, 0]

            if pieces.gradients is not None:
                gradients = pieces.gradients
                opt.gradients = to_numpy(gradients)[0, 0]

            if pieces.interventions is not None:
                interventions = pieces.interventions
                opt.interventions = to_numpy(interventions)[0]

            if self.embedder is not None:
                multipliers = attributor.doi._wrap_public_get_activation_multiplier(
                    activation=embeddings, model_inputs=inputs
                )
                opt.multipliers = to_numpy(multipliers)[0]

        # print(opt)

        return opt

    def tokens_stability(
        self,
        texts1: Iterable[str],
        texts2: Optional[Iterable[str]] = None,
        attributor: Optional[AttributionMethod] = None,
        show_id: bool = False,
        show_doi: bool = False,
        show_scale: bool = False,
        show_text: bool = False,
        extra_model_inputs: dict={}
    ):
        """
        Visualize decoded token from sentence pairs. Shows pairs side-by-side
        and highlights differences in them.
        """

        B = get_backend()

        if show_doi:
            if attributor is None:
                raise ValueError(
                    "show_doi requires attributor with doi to be given."
                )

            if not attributor._return_grads or not attributor._return_doi:
                raise ValueError(
                    "show_doi requires attributor to be configured with return_grads, return_doi."
                )

            if self.embedder is None or self.embeddings is None:
                raise ValueError(
                    "show_doi requires embedder and embeddings to be provided to NLP constructor."
                )

        if self.tokenize is None:
            raise ValueError("tokenize not provided to NLP visualizer.")

        textss = [texts1]
        if texts2 is not None:
            textss.append(texts2)

        opts = [
            self._get_optionals(texts, attributor=attributor, extra_model_inputs=extra_model_inputs)
            for texts in textss
        ]

        # Accumulate total output here.
        content = []

        # Include a scale if an attributor was provided.
        if attributor is not None and show_scale:
            content += [
                self.token_attribution_scale(),
                self.output.linebreak(),
                self.output.linebreak()
            ]

        # For each sentence,
        for i in range(len(texts1)):

            opt0 = opts[0].for_text_index(i)
            if len(textss) > 1:
                opt1 = opts[1].for_text_index(i)

            # Accumulate per-sentence output here.
            aline = []

            # If there are multiple texts, determine parts that differ to highlight.
            if len(textss) > 1:
                highlights = list(
                    arrays_different(opt0.input_ids, opt1.input_ids)
                )
            else:
                highlights = [False] * len(opt0.input_ids)

            # Add the visualization of the sentence.
            aline.append(
                self._tokens_stability_instance(
                    opt0,
                    index=i,
                    show_id=show_id,
                    show_doi=show_doi,
                    show_text=show_text,
                    highlights=highlights,
                    attributor=attributor
                )
            )

            # Add the visualization of its pair of multiple texts were provided.
            if len(textss) > 1:
                aline.append(
                    self._tokens_stability_instance(
                        opt1,
                        index=i,
                        show_id=show_id,
                        show_doi=show_doi,
                        show_text=show_text,
                        highlights=highlights,
                        attributor=attributor
                    )
                )

            # Add the accumulated elements to the final output.
            content.append(self.output.line(self.output.concat(*aline)))

        # Concat/render the entire content.
        return self.output.render(self.output.concat(*content))

    def tokens(
        self,
        texts,
        attributor: AttributionMethod = None,
        show_id: bool = False,
        show_doi: bool = False,
        show_text: bool = False,
        show_scale: bool = False,
        extra_model_inputs: dict={}
    ):
        """Visualize a token-based input attribution."""

        return self.tokens_stability(
            texts1=texts,
            attributor=attributor,
            show_id=show_id,
            show_doi=show_doi,
            show_text=show_text,
            show_scale=show_scale,
            extra_model_inputs=extra_model_inputs
        )

    def _closest_token(self, emb):
        distances = self.embedding_distance(emb)
        closest = np.argsort(distances)
        return closest[0], distances[closest[0]]
